use chrono::Utc;
use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct FileWatcherState {
    pub recent_changes: VecDeque<FileChange>,
    pub auto_lint: bool,
    pub last_lint_time: Instant,
    pub active_lint_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FileChange {
    pub path: String,
    pub filename: String,
    pub timestamp: String,
}

impl FileWatcherState {
    pub fn new() -> Self {
        Self {
            recent_changes: VecDeque::with_capacity(20),
            auto_lint: false,
            last_lint_time: Instant::now(),
            active_lint_error: None,
        }
    }

    pub fn push(&mut self, change: FileChange) {
        if self.recent_changes.len() >= 20 {
            self.recent_changes.pop_front();
        }
        self.recent_changes.push_back(change);
    }

    /// Keywords from recent file changes for search boosting.
    pub fn get_boost_keywords(&self) -> Vec<String> {
        let mut words = Vec::new();
        for c in &self.recent_changes {
            let stem = c.filename.split('.').next().unwrap_or(&c.filename);
            let mut current_word = String::new();
            for ch in stem.chars() {
                if ch.is_alphanumeric() {
                    if ch.is_uppercase() && !current_word.is_empty() {
                        words.push(current_word.clone());
                        current_word.clear();
                    }
                    current_word.push(ch);
                } else if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
            }
            if !current_word.is_empty() {
                words.push(current_word);
            }
        }
        words
    }
}

pub fn start_watcher(dir: &str) -> Option<Arc<Mutex<FileWatcherState>>> {
    let state = Arc::new(Mutex::new(FileWatcherState::new()));
    let state_clone = state.clone();
    let state_linter = state.clone();
    let dir_path = PathBuf::from(dir);
    let dir_str = dir.to_string();

    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = match notify::recommended_watcher(move |res: Result<Event, _>| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        }) {
            Ok(w) => w,
            Err(_) => return,
        };

        if watcher.watch(&dir_path, RecursiveMode::Recursive).is_err() {
            return;
        }

        for event in rx {
            if !matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                continue;
            }
            for path in &event.paths {
                let path_str = path.to_string_lossy();
                // Skip .git, node_modules, target, hidden files
                if path_str.contains("/.")
                    || path_str.contains("/node_modules/")
                    || path_str.contains("/target/")
                {
                    continue;
                }
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                if filename.is_empty() {
                    continue;
                }

                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if !["rs", "ts", "svelte", "py", "js", "go", "tsx", "jsx", "md"].contains(&ext)
                    {
                        continue;
                    }
                }

                if let Ok(mut s) = state_clone.lock() {
                    s.push(FileChange {
                        path: path_str.to_string(),
                        filename: filename.clone(),
                        timestamp: Utc::now().to_rfc3339(),
                    });
                }
            }
        }
    });

    // Background linter thread — opens one DB connection for its lifetime
    std::thread::spawn(move || {
        let local_db = crate::db::Database::open().ok();

        loop {
            std::thread::sleep(Duration::from_secs(5));

            let should_lint = if let Ok(mut s) = state_linter.lock() {
                if !s.auto_lint {
                    false
                } else if s.last_lint_time.elapsed() > Duration::from_secs(4) {
                    s.last_lint_time = Instant::now();
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if !should_lint {
                continue;
            }

            let dir_p = Path::new(&dir_str);
            let mut cmd = None;
            if dir_p.join("Cargo.toml").exists() {
                cmd = Some(("cargo", vec!["check"]));
            } else if dir_p.join("package.json").exists() {
                if dir_p.join("svelte.config.js").exists() {
                    cmd = Some(("npx", vec!["svelte-check"]));
                } else if dir_p.join("tsconfig.json").exists() {
                    cmd = Some(("npx", vec!["tsc", "--noEmit"]));
                }
            }

            let Some((program, args)) = cmd else {
                continue;
            };
            let Ok(output) = std::process::Command::new(program)
                .args(&args)
                .current_dir(&dir_str)
                .output()
            else {
                continue;
            };

            let is_error = !output.status.success();
            let mut error_msg = String::from_utf8_lossy(&output.stderr).to_string();
            if error_msg.trim().is_empty() {
                error_msg = String::from_utf8_lossy(&output.stdout).to_string();
            }

            let tags = vec!["auto-linter".to_string(), "bug".to_string()];

            if is_error {
                let content = format!(
                    "Auto-Linter Error:\nCommand: {} {:?}\nError:\n{}",
                    program, args, error_msg
                );

                let should_save = if let Ok(mut s) = state_linter.lock() {
                    if s.active_lint_error.as_deref() != Some(&content) {
                        s.active_lint_error = Some(content.clone());
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if should_save {
                    if let Some(ref db) = local_db {
                        let _ = db.add_memory(
                            &content,
                            "bug",
                            None,
                            &tags,
                            "auto-linter",
                            5,
                            None,
                            None,
                            &crate::db::MemoryScope::default(),
                        );
                    }
                }
            } else {
                let cleared = if let Ok(mut s) = state_linter.lock() {
                    if s.active_lint_error.is_some() {
                        s.active_lint_error = None;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if cleared {
                    if let Some(ref db) = local_db {
                        if let Ok(results) = db.search(
                            "Auto-Linter Error",
                            10,
                            None,
                            Some("bug"),
                            Some(&tags),
                            None,
                        ) {
                            for res in results {
                                let _ = db.delete_memory(&res.memory.id);
                            }
                        }
                    }
                }
            }
        }
    });

    Some(state)
}
