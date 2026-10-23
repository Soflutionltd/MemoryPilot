//! Reused Postgres sessions for request handling. A session only carries
//! the `memorypilot_app` role (set once when it opens); the tenant is set
//! per transaction, so a session moves between scopes safely.

use postgres::Client;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::{pg_error, MasterKey, PgConnection, Scope};

/// Last expiry pass per scope, shared by the stores of one pool.
pub(crate) type ExpiryClock = Arc<Mutex<HashMap<Scope, Instant>>>;

#[derive(Clone)]
pub struct PgPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    connection: PgConnection,
    idle: Mutex<Vec<Client>>,
    max_idle: usize,
    expiries: ExpiryClock,
    master: Option<MasterKey>,
}

impl PgPool {
    /// Keeps up to `max_idle` sessions open between requests.
    pub fn new(connection: PgConnection, max_idle: usize) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                connection,
                idle: Mutex::new(Vec::new()),
                max_idle,
                expiries: ExpiryClock::default(),
                master: None,
            }),
        }
    }

    /// The master key that unwraps each tenant's data key (credentials).
    pub fn with_master_key(self, key: MasterKey) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                connection: self.inner.connection.clone(),
                idle: Mutex::new(Vec::new()),
                max_idle: self.inner.max_idle,
                expiries: Arc::clone(&self.inner.expiries),
                master: Some(key),
            }),
        }
    }

    pub(crate) fn master_key(&self) -> Option<&MasterKey> {
        self.inner.master.as_ref()
    }

    pub(crate) fn checkout(&self) -> Result<PooledClient, String> {
        let reused = {
            let mut idle = self
                .inner
                .idle
                .lock()
                .map_err(|_| "Postgres pool lock poisoned".to_string())?;
            std::iter::from_fn(|| idle.pop()).find(|client| !client.is_closed())
        };
        let client = match reused {
            Some(client) => client,
            None => app_session(&self.inner.connection)?,
        };
        Ok(PooledClient {
            client: Some(client),
            home: Some(Arc::clone(&self.inner)),
        })
    }

    pub(crate) fn expiries(&self) -> ExpiryClock {
        Arc::clone(&self.inner.expiries)
    }

    pub fn idle_sessions(&self) -> usize {
        self.inner.idle.lock().map(|idle| idle.len()).unwrap_or(0)
    }
}

/// A new session as the application role: row level security applies.
pub(crate) fn app_session(connection: &PgConnection) -> Result<Client, String> {
    let mut client = connection.connect()?;
    client
        .batch_execute("SET ROLE memorypilot_app")
        .map_err(pg_error)?;
    Ok(client)
}

/// A session that goes back to its pool when dropped, unless it broke.
pub(crate) struct PooledClient {
    client: Option<Client>,
    home: Option<Arc<PoolInner>>,
}

impl PooledClient {
    pub(crate) fn standalone(client: Client) -> Self {
        Self {
            client: Some(client),
            home: None,
        }
    }
}

impl Deref for PooledClient {
    type Target = Client;
    fn deref(&self) -> &Client {
        self.client.as_ref().expect("client present until drop")
    }
}

impl DerefMut for PooledClient {
    fn deref_mut(&mut self) -> &mut Client {
        self.client.as_mut().expect("client present until drop")
    }
}

impl Drop for PooledClient {
    fn drop(&mut self) {
        let (Some(client), Some(home)) = (self.client.take(), self.home.take()) else {
            return;
        };
        if client.is_closed() {
            return;
        }
        let Ok(mut idle) = home.idle.lock() else {
            return;
        };
        if idle.len() < home.max_idle {
            idle.push(client);
        }
    }
}
