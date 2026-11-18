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
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

const MIRROR_REPO: &str = "Soflutionltd/MemoryPilot";

/// A temporary file untouched for this long belongs to a download that
/// was killed: a live one writes to its file all along.
const STALE_PARTIAL: Duration = Duration::from_secs(3600);

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
    remove_stale_partials(&dir, SystemTime::now());
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
        // No network must cost the search seconds, not the whole download budget.
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
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

/// Removes the temporary files of `dir` left by a download that was
/// killed before it could clean up.
fn remove_stale_partials(dir: &Path, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.filter_map(Result::ok) {
        let partial = entry.path().extension().is_some_and(|extension| extension.to_string_lossy().starts_with("part-"));
        let stale = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .is_ok_and(|modified| now.duration_since(modified).is_ok_and(|age| age >= STALE_PARTIAL));
        if partial && stale {
            // Best effort: a file that stays is only disk space.
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Checks the print of a file already in the cache and removes it when it
/// does not match, so that the next call fetches it again. `Ok(true)`: a
/// corrupt file was removed. Meant for when a model fails to load: hashing
/// 268 MB on every start is not worth it.
pub fn remove_if_corrupt(tag: &str, file: &MirroredFile) -> Result<bool, String> {
    let path = crate::embedding::fastembed_cache_dir().join("memorypilot-mirror").join(tag).join(file.name);
    remove_if_corrupt_at(&path, file)
}

fn remove_if_corrupt_at(path: &Path, file: &MirroredFile) -> Result<bool, String> {
    let mut input = match std::fs::File::open(path) {
        Ok(input) => input,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("open {}: {}", path.display(), error)),
    };
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    loop {
        let read = input.read(&mut buffer).map_err(|error| format!("read {}: {}", path.display(), error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let print: String = hasher.finalize().iter().map(|byte| format!("{:02x}", byte)).collect();
    if print == file.sha256 {
        return Ok(false);
    }
    std::fs::remove_file(path).map_err(|error| format!("remove {}: {}", path.display(), error))?;
    Ok(true)
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
    fn a_corrupt_cached_file_is_removed_and_a_sound_one_kept() {
        let dir = std::env::temp_dir().join(format!("memorypilot-mirror-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(HELLO.name);
        assert_eq!(remove_if_corrupt_at(&path, &HELLO), Ok(false), "nothing cached");
        std::fs::write(&path, b"hello").expect("write");
        assert_eq!(remove_if_corrupt_at(&path, &HELLO), Ok(false));
        assert!(path.exists());
        // Same size, other content: what the size check alone lets through.
        std::fs::write(&path, b"jello").expect("write");
        assert!(is_in_place(&path, &HELLO));
        assert_eq!(remove_if_corrupt_at(&path, &HELLO), Ok(true));
        assert!(!path.exists());
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn only_old_temporary_files_are_swept() {
        let dir = std::env::temp_dir().join(format!("memorypilot-mirror-stale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let partial = dir.join("hello.part-1-0");
        let model = dir.join(HELLO.name);
        std::fs::write(&partial, b"he").expect("write");
        std::fs::write(&model, b"hello").expect("write");
        remove_stale_partials(&dir, SystemTime::now());
        assert!(partial.exists(), "a download in progress keeps its file");
        remove_stale_partials(&dir, SystemTime::now() + STALE_PARTIAL + Duration::from_secs(1));
        assert!(!partial.exists());
        assert!(model.exists(), "the model itself is never swept");
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
