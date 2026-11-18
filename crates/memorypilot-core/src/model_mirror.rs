//! Model files MemoryPilot hosts itself, as assets of a release of its
//! own repository, so that an installation does not depend on a
//! third-party repository staying online or unchanged.
//!
//! Each file is pinned by size and SHA-256. It is fetched from the mirror
//! first, from the publisher's pinned revision if the mirror cannot be
//! reached, and kept only when the print matches.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};

const MIRROR_REPO: &str = "Soflutionltd/MemoryPilot";

/// Tells apart the temporary files of the downloads of one process.
static DOWNLOAD_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One pinned file of a mirrored model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MirroredFile {
    /// File name, in the release and in the local cache.
    pub name: &'static str,
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the file.
    pub sha256: &'static str,
    /// The same file at the publisher's, at a pinned revision.
    pub upstream: &'static str,
}

/// Local path of `file` of the release `tag`, downloading it on first use.
pub fn mirrored_file(tag: &str, file: &MirroredFile) -> Result<PathBuf, String> {
    let dir = crate::embedding::fastembed_cache_dir().join("memorypilot-mirror").join(tag);
    let path = dir.join(file.name);
    if is_in_place(&path, file) {
        return Ok(path);
    }
    std::fs::create_dir_all(&dir).map_err(|error| format!("create {}: {}", dir.display(), error))?;
    let mirror = format!("https://github.com/{}/releases/download/{}/{}", MIRROR_REPO, tag, file.name);
    let mut failures = Vec::new();
    for url in [mirror.as_str(), file.upstream] {
        match download(url, file, &path) {
            Ok(()) => return Ok(path),
            Err(error) => failures.push(error),
        }
    }
    Err(format!("{} could not be fetched: {}", file.name, failures.join("; ")))
}

/// A file is renamed into place only once its print is checked, so its
/// presence at the pinned size is enough.
fn is_in_place(path: &Path, file: &MirroredFile) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.len() == file.bytes)
}

fn download(url: &str, file: &MirroredFile, path: &Path) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .timeout_global(Some(Duration::from_secs(1800)))
        .user_agent(concat!("MemoryPilot/", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|error| format!("{}: {}", url, error))?;
    // One byte past the expected size, so a longer file is seen as such.
    let reader = response.body_mut().with_config().limit(file.bytes + 1).reader();
    store_verified(reader, file, path).map_err(|error| format!("{}: {}", url, error))
}

/// Writes `reader` to `path` through a temporary file, kept only when the
/// size and the SHA-256 are the pinned ones. The temporary file is this
/// download's own: two servers starting together (one per editor) each
/// fetch their copy, and the second rename replaces an identical file.
fn store_verified(mut reader: impl Read, file: &MirroredFile, path: &Path) -> Result<(), String> {
    let partial = path.with_extension(format!(
        "part-{}-{}",
        std::process::id(),
        DOWNLOAD_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let outcome = (|| {
        let mut out = std::fs::File::create(&partial).map_err(|error| format!("create {}: {}", partial.display(), error))?;
        let mut hasher = Sha256::new();
        let mut written = 0u64;
        let mut buffer = vec![0u8; 1 << 16];
        loop {
            let read = reader.read(&mut buffer).map_err(|error| format!("read: {}", error))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            out.write_all(&buffer[..read]).map_err(|error| format!("write {}: {}", partial.display(), error))?;
            written += read as u64;
        }
        out.sync_all().map_err(|error| format!("sync {}: {}", partial.display(), error))?;
        if written != file.bytes {
            return Err(format!("{} bytes received, {} expected", written, file.bytes));
        }
        let print: String = hasher.finalize().iter().map(|byte| format!("{:02x}", byte)).collect();
        if print != file.sha256 {
            return Err(format!("SHA-256 {} does not match the pinned {}", print, file.sha256));
        }
        std::fs::rename(&partial, path).map_err(|error| format!("rename to {}: {}", path.display(), error))
    })();
    if outcome.is_err() {
        // Best effort: the partial file is never read back as the model.
        let _ = std::fs::remove_file(&partial);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SHA-256 of the five bytes `hello`.
    const HELLO: MirroredFile = MirroredFile {
        name: "hello.bin",
        bytes: 5,
        sha256: concat!("2cf24dba5fb0a30e26e83b2ac5b9e29e", "1b161e5c1fa7425e73043362938b9824"),
        upstream: "https://example.invalid/hello.bin",
    };

    #[test]
    fn a_file_is_kept_only_when_size_and_print_match() {
        let dir = std::env::temp_dir().join(format!("memorypilot-mirror-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(HELLO.name);

        let wrong = store_verified(&b"jello"[..], &HELLO, &path).unwrap_err();
        assert!(wrong.contains("does not match"), "{}", wrong);
        assert!(!path.exists());

        let short = store_verified(&b"hell"[..], &HELLO, &path).unwrap_err();
        assert!(short.contains("4 bytes received, 5 expected"), "{}", short);
        assert!(!path.exists());

        store_verified(&b"hello"[..], &HELLO, &path).expect("matching file");
        assert_eq!(std::fs::read(&path).expect("stored"), b"hello");
        assert!(is_in_place(&path, &HELLO));
        let left: Vec<_> = std::fs::read_dir(&dir).expect("list").filter_map(Result::ok).map(|entry| entry.file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from(HELLO.name)], "no temporary file is left behind");

        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn downloads_running_together_do_not_share_a_temporary_file() {
        let dir = std::env::temp_dir().join(format!("memorypilot-mirror-race-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(HELLO.name);
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || store_verified(&b"hello"[..], &HELLO, &path))
            })
            .collect();
        for worker in workers {
            worker.join().expect("thread").expect("every download succeeds");
        }
        assert_eq!(std::fs::read(&path).expect("stored"), b"hello");
        assert_eq!(std::fs::read_dir(&dir).expect("list").count(), 1);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
