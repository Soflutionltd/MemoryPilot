//! What a folder's Git metadata says about the project it belongs to: the
//! main repository behind a worktree, and the `origin` remote. Read from the
//! files under `.git`, without running `git`.

use std::path::{Path, PathBuf};

/// The Git directory holding `config` for `root`: `root/.git` for a
/// repository, the main repository's `.git` for a linked worktree.
fn common_git_dir(root: &Path) -> Option<PathBuf> {
    let dot_git = root.join(".git");
    if dot_git.is_dir() {
        return Some(dot_git);
    }
    let pointer = std::fs::read_to_string(&dot_git).ok()?;
    let git_dir = pointer.lines().find_map(|line| line.trim().strip_prefix("gitdir:"))?.trim();
    let git_dir = if Path::new(git_dir).is_absolute() {
        PathBuf::from(git_dir)
    } else {
        root.join(git_dir)
    };
    // A worktree's Git directory names the shared one in `commondir`; a
    // submodule's has no such file and is a repository of its own.
    match std::fs::read_to_string(git_dir.join("commondir")) {
        Ok(common) => {
            let common = common.trim();
            let common = if Path::new(common).is_absolute() {
                PathBuf::from(common)
            } else {
                git_dir.join(common)
            };
            std::fs::canonicalize(common).ok()
        }
        Err(_) => Some(git_dir),
    }
}

/// The main working tree of the repository `root` is a linked worktree of,
/// or `None` when `root` is not a worktree.
pub(super) fn main_worktree_root(root: &Path) -> Option<PathBuf> {
    if !root.join(".git").is_file() {
        return None;
    }
    let common = common_git_dir(root)?;
    if common.file_name().and_then(|name| name.to_str()) != Some(".git") {
        return None;
    }
    let main_root = common.parent()?.to_path_buf();
    let same = std::fs::canonicalize(root).ok().as_deref() == Some(main_root.as_path());
    (!same).then_some(main_root)
}

/// The normalized `origin` remote of the repository at `root`.
pub(super) fn origin_remote(root: &Path) -> Option<String> {
    let config = std::fs::read_to_string(common_git_dir(root)?.join("config")).ok()?;
    let mut in_origin = false;
    for line in config.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            let section: String = line.chars().filter(|character| !character.is_whitespace()).collect();
            in_origin = section.eq_ignore_ascii_case("[remote\"origin\"]");
            continue;
        }
        if !in_origin {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            if key.trim().eq_ignore_ascii_case("url") {
                return normalize_remote(value);
            }
        }
    }
    None
}

/// `host/owner/repository` in lower case, so the SSH and HTTPS spellings of
/// one remote compare equal. Credentials, port and `.git` are dropped: a
/// token written in a remote URL is never stored.
pub(super) fn normalize_remote(url: &str) -> Option<String> {
    let url = url.trim().trim_matches('"');
    if url.is_empty() {
        return None;
    }
    let (rest, has_scheme) = match url.split_once("://") {
        Some((_, rest)) => (rest, true),
        None => (url, false),
    };
    let (host_part, path) = if has_scheme {
        rest.split_once('/')?
    } else {
        // `git@host:owner/repository`; anything else is a local path.
        rest.split_once(':').filter(|(host, _)| !host.contains('/'))?
    };
    let host = host_part.rsplit_once('@').map(|(_, host)| host).unwrap_or(host_part);
    let host = if has_scheme { host.split_once(':').map(|(host, _)| host).unwrap_or(host) } else { host };
    let path = path.trim_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path).trim_end_matches('/');
    if host.is_empty() || path.is_empty() {
        return None;
    }
    Some(format!("{}/{}", host, path).to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_and_https_spellings_of_a_remote_are_equal() {
        let expected = Some("github.com/acme/widget".to_string());
        assert_eq!(normalize_remote("git@github.com:Acme/Widget.git"), expected);
        assert_eq!(normalize_remote("https://github.com/acme/widget"), expected);
        assert_eq!(normalize_remote("https://github.com/acme/widget.git/"), expected);
        assert_eq!(normalize_remote("ssh://git@github.com:22/acme/widget.git"), expected);
    }

    #[test]
    fn credentials_in_a_remote_are_never_kept() {
        let normalized = normalize_remote("https://someone:s3cr3t-token@github.com/acme/widget.git").expect("remote");
        assert_eq!(normalized, "github.com/acme/widget");
    }

    #[test]
    fn local_paths_and_empty_values_are_not_remotes() {
        assert_eq!(normalize_remote(""), None);
        assert_eq!(normalize_remote("/srv/git/widget.git"), None);
        assert_eq!(normalize_remote("../widget"), None);
    }
}
