//! How the server reaches Postgres: TLS required, with the server
//! certificate and host name verified (the equivalent of
//! `sslmode=verify-full`), or a plain connection for a database on the same
//! machine or private network.

use postgres::config::SslMode;
use postgres::{Client, Config, NoTls};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::CertificateDer;
use std::path::Path;
use std::sync::Arc;
use tokio_postgres_rustls::MakeRustlsConnect;

#[derive(Clone)]
pub struct PgConnection {
    config: Config,
    tls: Option<MakeRustlsConnect>,
}

impl PgConnection {
    /// TLS required, verified against the Mozilla roots plus the
    /// certificates of `extra_ca` (PEM; for Supabase, the project's
    /// `prod-ca-2021.crt`). The URL's own `sslmode` is overridden: a server
    /// without TLS is refused, never used in clear.
    pub fn verified(url: &str, extra_ca: Option<&Path>) -> Result<Self, String> {
        let mut config = parse(url)?;
        config.ssl_mode(SslMode::Require);
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        if let Some(path) = extra_ca {
            let unreadable =
                |error: &dyn std::fmt::Display| format!("CA file {}: {}", path.display(), error);
            let mut added = 0usize;
            for certificate in CertificateDer::pem_file_iter(path).map_err(|e| unreadable(&e))? {
                roots
                    .add(certificate.map_err(|e| unreadable(&e))?)
                    .map_err(|e| unreadable(&e))?;
                added += 1;
            }
            if added == 0 {
                return Err(unreadable(&"no certificate"));
            }
        }
        let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("TLS: {}", e))?
        .with_root_certificates(roots)
        .with_no_client_auth();
        Ok(Self {
            config,
            tls: Some(MakeRustlsConnect::new(tls)),
        })
    }

    /// Unencrypted. Only for a database on this machine or a private
    /// network (local development, tests, a Compose network).
    pub fn plain(url: &str) -> Result<Self, String> {
        let mut config = parse(url)?;
        config.ssl_mode(SslMode::Disable);
        Ok(Self { config, tls: None })
    }

    pub fn is_encrypted(&self) -> bool {
        self.tls.is_some()
    }

    pub(crate) fn connect(&self) -> Result<Client, String> {
        let connected = match &self.tls {
            Some(tls) => self.config.connect(tls.clone()),
            None => self.config.connect(NoTls),
        };
        connected.map_err(|error| {
            // The handshake failure (unknown issuer, wrong host name) is in the source chain.
            let mut message = crate::pg_message(&error);
            let mut source = std::error::Error::source(&error);
            while let Some(cause) = source {
                message.push_str(": ");
                message.push_str(&cause.to_string());
                source = cause.source();
            }
            message
        })
    }
}

fn parse(url: &str) -> Result<Config, String> {
    url.parse::<Config>().map_err(|_| {
        "Postgres URL is not valid (postgresql://user:password@host:port/database)".to_string()
    })
}
