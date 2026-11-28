//! Model files fetched from their publisher's repository at one pinned
//! revision, each checked against a pinned size and SHA-256.
//!
//! A repository followed on its main branch can change under an
//! installation, and a file damaged in the cache (a download cut short, a
//! disk error) made every start fail without repairing itself. A file is
//! hashed once, when it enters the cache or is first seen there; a marker
//! next to it then spares the 250 MB read at every start. When the model
//! fails to load, `repair` hashes again and removes what does not match,
//! so that the next load downloads it.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// One pinned file of a model repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedFile {
    /// Path inside the repository.
    pub path: &'static str,
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the file.
    pub sha256: &'static str,
}

/// A model repository at one revision.
#[derive(Clone, Copy, Debug)]
pub struct PinnedRepo {
    pub repo: &'static str,
    /// Full commit hash.
    pub revision: &'static str,
}

impl PinnedRepo {
    /// Where the Hugging Face cache layout keeps `file` of this revision.
    fn cached(&self, cache: &Path, file: &PinnedFile) -> PathBuf {
        cache
            .join(format!("models--{}", self.repo.replace('/', "--")))
            .join("snapshots")
            .join(self.revision)
            .join(file.path)
    }

    /// Whether `file` is in the cache, at its pinned size.
    pub fn is_cached(&self, cache: &Path, file: &PinnedFile) -> bool {
        has_size(&self.cached(cache, file), file.bytes)
    }

    /// Local path of `file`, checked: downloaded on first use, and once
    /// more when the copy in the cache does not match its pin.
    pub fn file(&self, cache: &Path, file: &PinnedFile, show_progress: bool) -> Result<PathBuf, String> {
        let path = self.cached(cache, file);
        if path.exists() {
            match verify(&path, file) {
                Ok(()) => return Ok(path),
                Err(error) => eprintln!("[MemoryPilot] {}", error),
            }
        }
        let fetched = self.download(cache, file, show_progress)?;
        if fetched != path {
            return Err(format!(
                "{}/{}: downloaded to {}, expected at {}",
                self.repo,
                file.path,
                fetched.display(),
                path.display()
            ));
        }
        verify(&path, file)?;
        Ok(path)
    }

    fn download(&self, cache: &Path, file: &PinnedFile, show_progress: bool) -> Result<PathBuf, String> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache.to_path_buf())
            .with_progress(show_progress)
            .build()
            .map_err(|error| format!("Hugging Face API init: {}", error))?;
        let repo = hf_hub::Repo::with_revision(self.repo.to_string(), hf_hub::RepoType::Model, self.revision.to_string());
        api.repo(repo)
            .get(file.path)
            .map_err(|error| format!("download {}/{} at {}: {}", self.repo, file.path, &self.revision[..12], error))
    }

    /// Hashes `files` again, whatever their markers say, and removes those
    /// that do not match. Returns how many were removed.
    pub fn repair(&self, cache: &Path, files: &[PinnedFile]) -> usize {
        let mut removed = 0;
        for file in files {
            let path = self.cached(cache, file);
            if !path.exists() {
                continue;
            }
            forget(&path);
            if verify(&path, file).is_err() {
                removed += 1;
            }
        }
        removed
    }
}

fn has_size(path: &Path, bytes: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.len() == bytes)
}

/// Written next to a file once its print was checked, holding that print.
fn marker(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".sha256-checked");
    PathBuf::from(name)
}

fn forget(path: &Path) {
    // Best effort: a marker that stays is checked against the size and
    // the pinned print before it is believed.
    let _ = std::fs::remove_file(marker(path));
}

/// Checks `path` against its pin. A file that does not match is removed
/// (with the cache blob it points to), so that the next call downloads it.
fn verify(path: &Path, file: &PinnedFile) -> Result<(), String> {
    let checked = std::fs::read_to_string(marker(path)).is_ok_and(|print| print.trim() == file.sha256);
    if checked && has_size(path, file.bytes) {
        return Ok(());
    }
    let found = if has_size(path, file.bytes) { sha256_of(path)? } else { String::new() };
    if found == file.sha256 {
        std::fs::write(marker(path), file.sha256).map_err(|error| format!("{}: {}", marker(path).display(), error))?;
        return Ok(());
    }
    let size = std::fs::metadata(path).map(|metadata| metadata.len()).unwrap_or(0);
    remove(path)?;
    Err(format!(
        "{} does not match its pinned print ({} bytes, {} expected{}): removed, it is downloaded again at the next load",
        path.display(),
        size,
        file.bytes,
        if found.is_empty() { String::new() } else { format!("; SHA-256 {}", found) }
    ))
}

/// Removes the file, the cache blob behind it when it is a link, and its marker.
fn remove(path: &Path) -> Result<(), String> {
    forget(path);
    let blob = std::fs::canonicalize(path).ok().filter(|target| target != path);
    std::fs::remove_file(path).map_err(|error| format!("remove {}: {}", path.display(), error))?;
    if let Some(blob) = blob {
        match std::fs::remove_file(&blob) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("remove {}: {}", blob.display(), error)),
        }
    }
    Ok(())
}

fn sha256_of(path: &Path) -> Result<String, String> {
    let mut input = std::fs::File::open(path).map_err(|error| format!("open {}: {}", path.display(), error))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    loop {
        let read = input.read(&mut buffer).map_err(|error| format!("read {}: {}", path.display(), error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().iter().map(|byte| format!("{:02x}", byte)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO: PinnedRepo = PinnedRepo { repo: "vendor/model", revision: "0123456789abcdef0123456789abcdef01234567" };
    /// SHA-256 of the five bytes `hello`.
    const HELLO: PinnedFile = PinnedFile {
        path: "onnx/hello.bin",
        bytes: 5,
        sha256: concat!("2cf24dba5fb0a30e26e83b2ac5b9e29e", "1b161e5c1fa7425e73043362938b9824"),
    };

    /// A cache laid out as the Hugging Face client does: the snapshot holds
    /// a link to a blob.
    fn cache_with(name: &str, content: &[u8]) -> (PathBuf, PathBuf, PathBuf) {
        let cache = std::env::temp_dir().join(format!("memorypilot-pin-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        let path = REPO.cached(&cache, &HELLO);
        std::fs::create_dir_all(path.parent().expect("snapshot folder")).expect("snapshot");
        let blobs = cache.join("models--vendor--model").join("blobs");
        std::fs::create_dir_all(&blobs).expect("blobs");
        let blob = blobs.join("blob");
        std::fs::write(&blob, content).expect("blob");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, &path).expect("link");
        #[cfg(not(unix))]
        std::fs::copy(&blob, &path).expect("copy");
        (cache, path, blob)
    }

    #[test]
    fn a_cached_file_is_hashed_once_then_trusted_by_its_marker() {
        let (cache, path, blob) = cache_with("once", b"hello");
        assert!(REPO.is_cached(&cache, &HELLO));
        assert_eq!(REPO.file(&cache, &HELLO, false), Ok(path.clone()));
        assert_eq!(std::fs::read_to_string(marker(&path)).expect("marker"), HELLO.sha256);

        // Damaged afterwards at the same size: the marker is believed (no
        // 250 MB read at every start), a repair is not fooled.
        std::fs::write(&blob, b"jello").expect("damage");
        assert_eq!(REPO.file(&cache, &HELLO, false), Ok(path.clone()));
        assert_eq!(REPO.repair(&cache, &[HELLO]), 1);
        assert!(!path.exists() && !blob.exists() && !marker(&path).exists());
        assert!(!REPO.is_cached(&cache, &HELLO));
        assert_eq!(REPO.repair(&cache, &[HELLO]), 0, "nothing left to repair");
        let _ = std::fs::remove_dir_all(cache);
    }

    #[test]
    fn a_file_that_does_not_match_is_removed_with_its_blob() {
        let (cache, path, blob) = cache_with("wrong", b"jello");
        let error = verify(&path, &HELLO).expect_err("wrong print");
        assert!(error.contains("does not match its pinned print"), "{}", error);
        assert!(!path.exists() && !blob.exists());

        // A download cut short: the size alone tells, nothing is hashed.
        let (cache_short, path, blob) = cache_with("short", b"hell");
        assert!(!REPO.is_cached(&cache_short, &HELLO));
        let error = verify(&path, &HELLO).expect_err("short file");
        assert!(error.contains("4 bytes, 5 expected"), "{}", error);
        assert!(!path.exists() && !blob.exists());

        // A marker left by an earlier file does not vouch for another size.
        let (cache_marked, path, _) = cache_with("marked", b"hell");
        std::fs::write(marker(&path), HELLO.sha256).expect("marker");
        assert!(verify(&path, &HELLO).is_err());
        assert!(!marker(&path).exists());
        for cache in [cache, cache_short, cache_marked] {
            let _ = std::fs::remove_dir_all(cache);
        }
    }

    #[test]
    fn a_good_file_survives_a_repair() {
        let (cache, path, blob) = cache_with("good", b"hello");
        assert_eq!(REPO.repair(&cache, &[HELLO]), 0);
        assert!(path.exists() && blob.exists());
        assert_eq!(std::fs::read_to_string(marker(&path)).expect("marker"), HELLO.sha256);
        let _ = std::fs::remove_dir_all(cache);
    }
}
