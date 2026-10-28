//! Automatic updates from the GitHub releases.
//!
//! Every release publishes `latest.json` (version, SHA-256 of each archive,
//! progressive rollout, pause) signed with minisign. The secret key lives in
//! 1Password only; the public key is compiled in below. A file whose digest is
//! not in a signed manifest is never installed, and an older version is never
//! installed over a newer one.
//!
//! A long-running MemoryPilot (stdio or HTTP server) checks two minutes after
//! it starts, then every six hours. Before switching, the downloaded binary
//! must report the expected version, carry the Soflution Developer ID
//! signature on macOS, and open a copy of the user's database with its own
//! schema upgrades (`--self-test`). The database is first copied to
//! `~/.MemoryPilot/backups/`, and the previous binary stays next to the new one
//! (`MemoryPilot.old`).
//!
//! Every running instance (Cursor, Claude, McpHub, the HTTP server) notices the
//! new binary and replaces itself in place (`exec`: same process, same pipes)
//! as soon as it is idle, so the client keeps talking to it without a restart.
//! A version that crashes three times at startup before serving a request is
//! rolled back and never offered again on this machine.

use std::collections::HashMap;
use std::fmt::Display;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const REPO: &str = "Soflutionltd/MemoryPilot";
const UPDATE_PUBLIC_KEY: &str = "RWTJP+WH7HDqnwQsHDDjjFF2hNs2NcRxhNa10yeSkjB5qrYuDdqkIE7M";
const VERSION: &str = env!("CARGO_PKG_VERSION");

const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const MAX_ASSET_BYTES: u64 = 200 * 1024 * 1024;
const FIRST_CHECK_DELAY: Duration = Duration::from_secs(120);
const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 3600);
/// Uptime after which a freshly installed version counts as healthy.
const PROBATION: Duration = Duration::from_secs(120);
/// Startup crashes of a new version tolerated before going back.
const MAX_CRASHED_BOOTS: usize = 3;
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(120);
const BACKUPS_KEPT: usize = 2;
const LOCK_STALE_AFTER: Duration = Duration::from_secs(15 * 60);

static EXECUTABLE: OnceLock<Option<Executable>> = OnceLock::new();
static RESTART_FAILED: AtomicBool = AtomicBool::new(false);
static HEALTHY: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Deserialize)]
struct Manifest {
    version: String,
    rollout: Rollout,
    #[serde(default)]
    paused: bool,
    assets: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct Rollout {
    /// Share of installations served before `full_at` (0 to 100).
    percent: u8,
    /// Unix time from which every installation is served.
    full_at: u64,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Trigger {
    Manual,
    Automatic,
}

#[derive(Debug)]
enum Outcome {
    UpToDate,
    Paused(String),
    /// Outside the share served so far; served at `full_at` at the latest.
    NotYet { version: String, full_at: u64 },
    /// Rolled back on this machine after crashing at startup.
    Rejected(String),
    Installed { version: String, backup: Option<PathBuf> },
}

type Fingerprint = (u64, SystemTime);

/// The binary this process was started from, resolved once at startup: on
/// Linux `current_exe()` points to a deleted file once it has been replaced.
struct Executable {
    path: PathBuf,
    fingerprint: Option<Fingerprint>,
}

fn executable() -> Option<&'static Executable> {
    EXECUTABLE
        .get_or_init(|| {
            let path = std::env::current_exe().ok()?;
            let path = fs::canonicalize(&path).unwrap_or(path);
            let fingerprint = fingerprint(&path);
            Some(Executable { path, fingerprint })
        })
        .as_ref()
}

fn fingerprint(path: &Path) -> Option<Fingerprint> {
    let meta = fs::metadata(path).ok()?;
    Some((meta.len(), meta.modified().ok()?))
}

fn log(message: impl Display) {
    eprintln!("[MemoryPilot][update] {}", message);
}

fn data_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".MemoryPilot"))
}

fn update_dir() -> Option<PathBuf> {
    data_dir().map(|dir| dir.join("update"))
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

/// First call of a long-running server mode, before the database opens:
/// records the binary this process runs from and handles the probation of a
/// freshly installed version (rollback after repeated startup crashes).
pub fn startup_guard() {
    let Some(exe) = executable() else { return };
    let Some(dir) = update_dir() else { return };
    let mut state = read_state(&dir);
    let action = on_boot(&mut state, VERSION, crashed_boots(&dir, VERSION));
    if let Err(error) = write_state(&dir, &state) {
        log(error);
    }
    match action {
        BootAction::Continue => {}
        BootAction::Probation => {
            if let Err(error) = create_boot_marker(&dir) {
                log(error);
            }
        }
        BootAction::RollBack => {
            remove_boot_markers(&dir, VERSION);
            log(format!(
                "v{} crashed {} times at startup: restoring the previous version",
                VERSION, MAX_CRASHED_BOOTS
            ));
            let old = old_binary(&exe.path);
            if let Err(error) = fs::rename(&old, &exe.path) {
                log(format!("cannot restore {}: {}", old.display(), error));
                return;
            }
            restart_process(&exe.path);
        }
    }
}

/// Background work of a long-running server: probation confirmation by
/// uptime and, unless turned off, the update checks.
pub fn spawn_background() {
    std::thread::spawn(|| {
        std::thread::sleep(PROBATION);
        mark_healthy();
    });
    match automatic_updates_blocked() {
        Some(reason) => log(format!("automatic updates off: {}", reason)),
        None => {
            std::thread::spawn(check_loop);
        }
    }
}

fn check_loop() {
    let bucket = update_dir().map(|dir| install_bucket(&dir)).unwrap_or(0);
    std::thread::sleep(FIRST_CHECK_DELAY + Duration::from_secs(u64::from(bucket) * 6));
    loop {
        if !restart_pending() {
            match check_and_install(Trigger::Automatic) {
                Ok(Outcome::Installed { version, .. }) => log(format!("v{} installed", version)),
                Ok(Outcome::NotYet { version, full_at }) => log(format!(
                    "v{} available, served here from {} (Unix time) at the latest",
                    version, full_at
                )),
                Ok(Outcome::Paused(version)) => log(format!("rollout of v{} paused", version)),
                Ok(Outcome::Rejected(version)) => {
                    log(format!("v{} was rolled back on this machine, skipped", version))
                }
                Ok(Outcome::UpToDate) => {}
                Err(error) => log(format!("check failed: {}", error)),
            }
        }
        std::thread::sleep(CHECK_INTERVAL);
    }
}

/// True once the binary on disk differs from the one this process started
/// from: an update was installed and this process should switch to it.
pub fn restart_pending() -> bool {
    if RESTART_FAILED.load(Ordering::Relaxed) {
        return false;
    }
    let Some(exe) = executable() else { return false };
    exe.fingerprint.is_some() && exe.path.is_file() && fingerprint(&exe.path) != exe.fingerprint
}

/// Replaces this process with the binary on disk, same arguments, same open
/// stdin and stdout. Call only when no request is in progress. Returns only
/// if the switch failed, in which case this process keeps serving.
pub fn restart() {
    let Some(exe) = executable() else { return };
    mark_clean_exit();
    log(format!("new binary installed: v{} restarts in place", VERSION));
    restart_process(&exe.path);
    RESTART_FAILED.store(true, Ordering::Relaxed);
}

#[cfg(unix)]
fn restart_process(exe: &Path) {
    use std::os::unix::process::CommandExt;
    let error = Command::new(exe).args(std::env::args_os().skip(1)).exec();
    log(format!("restart failed, staying on v{}: {}", VERSION, error));
}

#[cfg(not(unix))]
fn restart_process(exe: &Path) {
    log(format!("{} is used from the next start", exe.display()));
}

/// Watches for an installed update in a server without a request loop of its
/// own (HTTP) and restarts once `is_idle` has held for two checks in a row.
pub fn spawn_idle_restarter(is_idle: impl Fn() -> bool + Send + 'static) {
    std::thread::spawn(move || {
        let mut idle_checks = 0;
        loop {
            std::thread::sleep(Duration::from_secs(2));
            if !restart_pending() {
                idle_checks = 0;
                continue;
            }
            idle_checks = if is_idle() { idle_checks + 1 } else { 0 };
            if idle_checks >= 2 {
                restart();
                if RESTART_FAILED.load(Ordering::Relaxed) {
                    return;
                }
            }
        }
    });
}

/// The process has served a request (or stayed up `PROBATION`): a version on
/// probation is confirmed and the previous binary removed.
pub fn mark_healthy() {
    if HEALTHY.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some(dir) = update_dir() else { return };
    let mut state = read_state(&dir);
    let Some(from) = state
        .pending
        .as_ref()
        .filter(|pending| pending.to == VERSION)
        .map(|pending| pending.from.clone())
    else {
        mark_clean_exit();
        return;
    };
    state.pending = None;
    if let Err(error) = write_state(&dir, &state) {
        log(error);
    }
    if let Some(exe) = executable() {
        let _ = fs::remove_file(old_binary(&exe.path));
    }
    remove_boot_markers(&dir, VERSION);
    log(format!("v{} confirmed (previous: v{})", VERSION, from));
}

/// Normal shutdown or switch: this process did not crash.
pub fn mark_clean_exit() {
    if let Some(dir) = update_dir() {
        let _ = fs::remove_file(boot_marker(&dir, VERSION, std::process::id()));
    }
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

/// `MemoryPilot update [--check]`. Returns the process exit code.
pub fn run_cli(check_only: bool) -> i32 {
    println!("Checking for updates (current: v{})...", VERSION);
    if check_only {
        return match fetch_manifest() {
            Ok(manifest) if is_newer(&manifest.version, VERSION) => {
                println!("v{} is available. Run `MemoryPilot update` to install it.", manifest.version);
                0
            }
            Ok(_) => {
                println!("MemoryPilot is up to date (v{}).", VERSION);
                0
            }
            Err(error) => {
                eprintln!("✗ Update check failed: {}", error);
                1
            }
        };
    }
    match check_and_install(Trigger::Manual) {
        Ok(Outcome::UpToDate) => println!("MemoryPilot is up to date (v{}).", VERSION),
        Ok(Outcome::Paused(version)) => println!("The rollout of v{} is paused, nothing installed.", version),
        Ok(Outcome::NotYet { version, .. }) => println!("v{} is not served to this machine yet.", version),
        Ok(Outcome::Rejected(version)) => println!("v{} was rolled back on this machine, skipped.", version),
        Ok(Outcome::Installed { version, backup }) => {
            println!("Updated to v{}.", version);
            if let Some(backup) = backup {
                println!("Database copied to {} first.", backup.display());
            }
            println!("Running MemoryPilot servers switch to v{} as soon as they are idle.", version);
        }
        Err(error) => {
            eprintln!("✗ Update failed: {}", error);
            return 1;
        }
    }
    0
}

/// Why this binary must not replace itself, if it must not.
fn installation_blocked(exe: &Path, debug_build: bool) -> Option<&'static str> {
    if debug_build {
        return Some("debug build");
    }
    if exe.components().any(|component| component.as_os_str() == "target") {
        return Some("binary inside a cargo target directory");
    }
    let path = exe.to_string_lossy();
    if path.contains("/Cellar/") || path.contains("/homebrew/") || path.contains("/linuxbrew/") {
        return Some("installed by Homebrew: `brew upgrade memorypilot` updates it");
    }
    if asset_for(std::env::consts::OS, std::env::consts::ARCH).is_none() {
        return Some("no pre-built binary for this platform");
    }
    None
}

fn automatic_updates_blocked() -> Option<String> {
    if auto_update_disabled(std::env::var("MEMORYPILOT_AUTO_UPDATE").ok().as_deref()) {
        return Some("MEMORYPILOT_AUTO_UPDATE=0".to_string());
    }
    let Some(exe) = executable() else {
        return Some("executable path unknown".to_string());
    };
    installation_blocked(&exe.path, cfg!(debug_assertions)).map(str::to_string)
}

fn auto_update_disabled(raw: Option<&str>) -> bool {
    raw.map(|value| value.trim().to_ascii_lowercase())
        .is_some_and(|value| matches!(value.as_str(), "0" | "off" | "false" | "no"))
}

// ---------------------------------------------------------------------------
// Check and install
// ---------------------------------------------------------------------------

fn check_and_install(trigger: Trigger) -> Result<Outcome, String> {
    let exe = executable().ok_or("cannot locate the MemoryPilot executable")?;
    if let Some(reason) = installation_blocked(&exe.path, cfg!(debug_assertions)) {
        return Err(reason.to_string());
    }
    let data = data_dir().ok_or("cannot find the home directory")?;
    let dir = data.join("update");
    ensure_private_dir(&dir)?;
    let _lock = UpdateLock::acquire(&dir)?;

    let manifest = fetch_manifest()?;
    if !is_newer(&manifest.version, VERSION) {
        return Ok(Outcome::UpToDate);
    }
    if manifest.paused {
        return Ok(Outcome::Paused(manifest.version));
    }
    if trigger == Trigger::Automatic {
        if read_state(&dir).rejected.contains(&manifest.version) {
            return Ok(Outcome::Rejected(manifest.version));
        }
        if !rollout_allows(&manifest, install_bucket(&dir), unix_now()) {
            return Ok(Outcome::NotYet {
                version: manifest.version,
                full_at: manifest.rollout.full_at,
            });
        }
    }

    let asset = asset_for(std::env::consts::OS, std::env::consts::ARCH)
        .ok_or("no pre-built binary for this platform")?;
    let archive = download_verified(&manifest, asset)?;
    let work = PrivateDir::create()?;
    let archive_path = work.path.join(asset);
    fs::write(&archive_path, &archive).map_err(|error| format!("write {}: {}", archive_path.display(), error))?;
    let new_binary = extract(&work.path, &archive_path)?;
    #[cfg(target_os = "macos")]
    verify_developer_id(&new_binary)?;
    if !reports_version(&new_binary, &manifest.version) {
        return Err("the downloaded binary does not start or reports another version".into());
    }
    let backup = backup_database(&data, &manifest.version)?;
    self_test(&new_binary, backup.as_deref(), &work.path)?;
    install_binary(&exe.path, &new_binary)?;

    let mut state = read_state(&dir);
    state.pending = Some(Pending {
        from: VERSION.to_string(),
        to: manifest.version.clone(),
    });
    write_state(&dir, &state)?;
    prune_backups(&data.join("backups"), BACKUPS_KEPT);
    Ok(Outcome::Installed {
        version: manifest.version,
        backup,
    })
}

fn fetch_manifest() -> Result<Manifest, String> {
    let base = format!("https://github.com/{}/releases/latest/download", REPO);
    let body = fetch(&format!("{}/latest.json", base), MAX_MANIFEST_BYTES)?;
    let signature = fetch(&format!("{}/latest.json.minisig", base), MAX_MANIFEST_BYTES)?;
    let signature = String::from_utf8(signature).map_err(|_| "unreadable manifest signature".to_string())?;
    verify_manifest(&body, &signature, UPDATE_PUBLIC_KEY)
}

fn verify_manifest(body: &[u8], signature: &str, public_key: &str) -> Result<Manifest, String> {
    let key = minisign_verify::PublicKey::from_base64(public_key)
        .map_err(|error| format!("invalid update key: {}", error))?;
    let signature = minisign_verify::Signature::decode(signature)
        .map_err(|error| format!("malformed manifest signature: {}", error))?;
    key.verify(body, &signature, false)
        .map_err(|error| format!("manifest signature rejected: {}", error))?;
    let manifest: Manifest =
        serde_json::from_slice(body).map_err(|error| format!("malformed manifest: {}", error))?;
    if parse_version(&manifest.version).is_none() || manifest.rollout.percent > 100 {
        return Err("malformed manifest: version or rollout".into());
    }
    Ok(manifest)
}

fn rollout_allows(manifest: &Manifest, bucket: u8, now: u64) -> bool {
    now >= manifest.rollout.full_at || bucket < manifest.rollout.percent
}

/// Stable bucket 0 to 99 drawn from a local random id, never sent anywhere.
fn install_bucket(dir: &Path) -> u8 {
    let path = dir.join("install-id");
    let id = fs::read_to_string(&path)
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or_default();
            let id = format!("{:x}-{}", nanos, std::process::id());
            let _ = fs::write(&path, &id);
            id
        });
    bucket_of(&id)
}

fn bucket_of(id: &str) -> u8 {
    let digest = Sha256::digest(id.as_bytes());
    (u16::from_be_bytes([digest[0], digest[1]]) % 100) as u8
}

fn fetch(url: &str, limit: u64) -> Result<Vec<u8>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .timeout_global(Some(Duration::from_secs(600)))
        .user_agent(format!("MemoryPilot/{}", VERSION))
        .build()
        .into();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|error| format!("download failed ({}): {}", url, error))?;
    response
        .body_mut()
        .with_config()
        .limit(limit)
        .read_to_vec()
        .map_err(|error| format!("download failed ({}): {}", url, error))
}

fn download_verified(manifest: &Manifest, asset: &str) -> Result<Vec<u8>, String> {
    let expected = manifest
        .assets
        .get(asset)
        .ok_or_else(|| format!("{} is not listed in the signed manifest", asset))?;
    let url = format!(
        "https://github.com/{}/releases/download/v{}/{}",
        REPO, manifest.version, asset
    );
    let bytes = fetch(&url, MAX_ASSET_BYTES)?;
    check_sha256(&bytes, expected)?;
    Ok(bytes)
}

fn check_sha256(bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual: String = Sha256::digest(bytes).iter().map(|byte| format!("{:02x}", byte)).collect();
    if actual.eq_ignore_ascii_case(expected.trim()) {
        Ok(())
    } else {
        Err(format!("checksum mismatch (expected {}, got {})", expected, actual))
    }
}

fn extract(dir: &Path, archive: &Path) -> Result<PathBuf, String> {
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(dir)
        .arg("MemoryPilot")
        .stdin(Stdio::null())
        .status()
        .map_err(|error| format!("tar unavailable: {}", error))?;
    if !status.success() {
        return Err("extraction failed".into());
    }
    let binary = dir.join("MemoryPilot");
    if !binary.is_file() {
        return Err(format!("extracted binary not found at {}", binary.display()));
    }
    set_executable(&binary)?;
    Ok(binary)
}

/// Signed by Soflution (Developer ID, team TB8CC687M3): macOS would otherwise
/// ask again for the vault key in the Keychain, and a foreign binary is refused.
#[cfg(target_os = "macos")]
fn verify_developer_id(binary: &Path) -> Result<(), String> {
    let requirement = r#"=identifier "MemoryPilot" and anchor apple generic and certificate leaf[subject.OU] = "TB8CC687M3""#;
    let output = Command::new("/usr/bin/codesign")
        .args(["--verify", "--strict", "-R", requirement])
        .arg(binary)
        .output()
        .map_err(|error| format!("codesign unavailable: {}", error))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "the downloaded binary is not signed by Soflution: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn reports_version(binary: &Path, version: &str) -> bool {
    Command::new(binary)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map(|output| {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).trim() == format!("MemoryPilot v{}", version)
        })
        .unwrap_or(false)
}

/// Copy of `memory.db` taken before the new version touches it, kept in
/// `~/.MemoryPilot/backups/` (owner only). `None` on a fresh install.
fn backup_database(data: &Path, version: &str) -> Result<Option<PathBuf>, String> {
    let live = data.join("memory.db");
    if !live.is_file() {
        return Ok(None);
    }
    let backups = data.join("backups");
    ensure_private_dir(&backups)?;
    let target = backups.join(format!("memory-before-v{}.db", version));
    let _ = fs::remove_file(&target);
    let conn = rusqlite::Connection::open_with_flags(&live, rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE)
        .map_err(|error| format!("open {} for backup: {}", live.display(), error))?;
    conn.busy_timeout(Duration::from_secs(10))
        .map_err(|error| format!("backup busy timeout: {}", error))?;
    let target_text = target.to_string_lossy().into_owned();
    conn.execute("VACUUM INTO ?1", [target_text.as_str()])
        .map_err(|error| format!("database backup to {}: {}", target.display(), error))?;
    restrict_to_owner(&target)?;
    Ok(Some(target))
}

fn prune_backups(dir: &Path, keep: usize) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut backups: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with("memory-before-v") && name.ends_with(".db")
        })
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    backups.sort_by(|left, right| right.0.cmp(&left.0));
    for (_, path) in backups.into_iter().skip(keep) {
        let _ = fs::remove_file(path);
    }
}

/// The new binary opens a copy of the database with its own schema upgrades
/// before it may replace the running one.
fn self_test(new_binary: &Path, backup: Option<&Path>, work: &Path) -> Result<(), String> {
    let dir = work.join("self-test");
    fs::create_dir(&dir).map_err(|error| format!("self-test dir: {}", error))?;
    let db = dir.join("memory.db");
    if let Some(backup) = backup {
        fs::copy(backup, &db).map_err(|error| format!("self-test copy: {}", error))?;
    }
    let mut child = Command::new(new_binary)
        .arg("--self-test")
        .arg(&db)
        .env("MEMORYPILOT_VAULT", "off")
        .env("MEMORYPILOT_AUTO_UPDATE", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("self-test start: {}", error))?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() > SELF_TEST_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("the new version did not finish its self-test".into());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(error) => return Err(format!("self-test wait: {}", error)),
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("self-test output: {}", error))?;
    if output.status.success() && String::from_utf8_lossy(&output.stdout).contains("MemoryPilot self-test ok") {
        Ok(())
    } else {
        Err(format!(
            "the new version failed its self-test on a copy of the database: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

/// The previous binary stays as `MemoryPilot.old` until the new version is
/// confirmed healthy.
fn install_binary(exe: &Path, new_binary: &Path) -> Result<(), String> {
    let old = old_binary(exe);
    let staged = exe.with_extension("new");
    let _ = fs::remove_file(&staged);
    fs::copy(new_binary, &staged)
        .map_err(|error| format!("cannot write next to {}: {}", exe.display(), error))?;
    set_executable(&staged)?;
    let _ = fs::remove_file(&old);
    fs::rename(exe, &old).map_err(|error| {
        let _ = fs::remove_file(&staged);
        format!("cannot move the current binary: {}", error)
    })?;
    if let Err(error) = fs::rename(&staged, exe) {
        let _ = fs::rename(&old, exe);
        let _ = fs::remove_file(&staged);
        return Err(format!("cannot install the new binary: {}; previous version kept", error));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Probation and rollback
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
struct UpdateState {
    #[serde(default)]
    pending: Option<Pending>,
    /// Versions rolled back on this machine: never offered automatically again.
    #[serde(default)]
    rejected: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Pending {
    from: String,
    to: String,
}

#[derive(Debug, PartialEq, Eq)]
enum BootAction {
    Continue,
    Probation,
    RollBack,
}

fn on_boot(state: &mut UpdateState, running: &str, crashed: usize) -> BootAction {
    let Some(pending) = state.pending.as_ref() else {
        return BootAction::Continue;
    };
    if pending.to != running {
        // A process still on the previous version says nothing about the
        // new one; a newer version than the one on probation supersedes it.
        if is_newer(running, &pending.to) {
            state.pending = None;
        }
        return BootAction::Continue;
    }
    if crashed >= MAX_CRASHED_BOOTS {
        let rejected = pending.to.clone();
        state.pending = None;
        if !state.rejected.contains(&rejected) {
            state.rejected.push(rejected);
        }
        return BootAction::RollBack;
    }
    BootAction::Probation
}

fn old_binary(exe: &Path) -> PathBuf {
    exe.with_extension("old")
}

fn state_path(dir: &Path) -> PathBuf {
    dir.join("state.json")
}

fn read_state(dir: &Path) -> UpdateState {
    fs::read(state_path(dir))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn write_state(dir: &Path, state: &UpdateState) -> Result<(), String> {
    let path = state_path(dir);
    if *state == UpdateState::default() {
        let _ = fs::remove_file(&path);
        return Ok(());
    }
    ensure_private_dir(dir)?;
    let text = serde_json::to_vec_pretty(state).map_err(|error| error.to_string())?;
    let tmp = dir.join(format!("state.json.{}", std::process::id()));
    fs::write(&tmp, text).map_err(|error| format!("{}: {}", tmp.display(), error))?;
    fs::rename(&tmp, &path).map_err(|error| format!("{}: {}", path.display(), error))
}

/// One empty file per process running a version on probation, removed when
/// the process confirms the version or exits normally: files left by dead
/// processes are startup crashes.
fn boot_marker(dir: &Path, version: &str, pid: u32) -> PathBuf {
    dir.join("boots").join(format!("{}.{}", version, pid))
}

fn create_boot_marker(dir: &Path) -> Result<(), String> {
    let marker = boot_marker(dir, VERSION, std::process::id());
    if let Some(parent) = marker.parent() {
        ensure_private_dir(parent)?;
    }
    fs::write(&marker, b"").map_err(|error| format!("{}: {}", marker.display(), error))
}

fn marker_pids(dir: &Path, version: &str) -> Vec<u32> {
    let prefix = format!("{}.", version);
    let Ok(entries) = fs::read_dir(dir.join("boots")) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.strip_prefix(&prefix)?.parse::<u32>().ok()
        })
        .collect()
}

fn crashed_boots(dir: &Path, version: &str) -> usize {
    let own = std::process::id();
    marker_pids(dir, version)
        .into_iter()
        .filter(|pid| *pid != own && !process_alive(*pid))
        .count()
}

fn remove_boot_markers(dir: &Path, version: &str) {
    for pid in marker_pids(dir, version) {
        let _ = fs::remove_file(boot_marker(dir, version, pid));
    }
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    let Some(pid) = i32::try_from(pid).ok().and_then(rustix::process::Pid::from_raw) else {
        return false;
    };
    match rustix::process::test_kill_process(pid) {
        Ok(()) => true,
        Err(error) => error == rustix::io::Errno::PERM,
    }
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assets published by `.github/workflows/release.yml` (Linux) and
/// `scripts/release-macos.sh` (Apple Silicon, signed and notarized).
fn asset_for(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("macos", "aarch64") => Some("MemoryPilot-aarch64-apple-darwin.tar.gz"),
        ("linux", "x86_64") => Some("MemoryPilot-x86_64-unknown-linux-gnu.tar.gz"),
        ("linux", "aarch64") => Some("MemoryPilot-aarch64-unknown-linux-gnu.tar.gz"),
        _ => None,
    }
}

fn parse_version(version: &str) -> Option<Vec<u64>> {
    version
        .trim_start_matches('v')
        .split('.')
        .map(|part| part.parse().ok())
        .collect()
}

fn is_newer(candidate: &str, current: &str) -> bool {
    match (parse_version(candidate), parse_version(current)) {
        (Some(candidate), Some(current)) => candidate > current,
        _ => false,
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).map_err(|error| error.to_string())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn restrict_to_owner(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|error| error.to_string())
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn ensure_private_dir(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|error| format!("cannot create {}: {}", path.display(), error))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("{}: {}", path.display(), error))?;
    }
    Ok(())
}

/// One update at a time across every MemoryPilot process of the machine.
struct UpdateLock {
    path: PathBuf,
}

impl UpdateLock {
    fn acquire(dir: &Path) -> Result<Self, String> {
        let path = dir.join("update.lock");
        let stale = fs::metadata(&path)
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age > LOCK_STALE_AFTER);
        if stale {
            let _ = fs::remove_file(&path);
        }
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map(|_| Self { path })
            .map_err(|_| "another update is already running".to_string())
    }
}

impl Drop for UpdateLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Temporary directory of one update (exclusive creation, owner only): another
/// user of the machine cannot drop the archive or the binary in our place.
struct PrivateDir {
    path: PathBuf,
}

impl PrivateDir {
    fn create() -> Result<Self, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default();
        let path = std::env::temp_dir().join(format!("memorypilot-update-{}-{:x}", std::process::id(), nanos));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(&path)
            .map_err(|error| format!("cannot create {}: {}", path.display(), error))?;
        Ok(Self { path })
    }
}

impl Drop for PrivateDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/update-manifest.json");
    const FIXTURE_SIG: &str = include_str!("../tests/fixtures/update-manifest.json.minisig");
    const FIXTURE_KEY: &str = include_str!("../tests/fixtures/update-test-key.pub");
    const DARWIN: &str = "MemoryPilot-aarch64-apple-darwin.tar.gz";

    fn fixture() -> Manifest {
        verify_manifest(FIXTURE.as_bytes(), FIXTURE_SIG, FIXTURE_KEY.trim()).unwrap()
    }

    #[test]
    fn never_goes_back() {
        assert!(is_newer("4.12.0", "4.11.1"));
        assert!(is_newer("5.0.0", "4.11.1"));
        assert!(is_newer("4.11.10", "4.11.9"));
        assert!(!is_newer("4.11.1", "4.11.1"));
        assert!(!is_newer("4.11.0", "4.11.1"));
        assert!(!is_newer("garbage", "4.11.1"));
    }

    #[test]
    fn signed_manifest_accepted() {
        let manifest = fixture();
        assert_eq!(manifest.version, "9.0.0");
        assert_eq!(manifest.rollout.percent, 10);
        assert_eq!(manifest.assets.len(), 3);
        check_sha256(b"darwin", &manifest.assets[DARWIN]).unwrap();
        assert!(check_sha256(b"tampered", &manifest.assets[DARWIN]).is_err());
    }

    #[test]
    fn tampered_manifest_or_other_key_refused() {
        let tampered = FIXTURE.replace("\"percent\": 10", "\"percent\": 100");
        assert!(verify_manifest(tampered.as_bytes(), FIXTURE_SIG, FIXTURE_KEY.trim()).is_err());
        // The release key does not validate a test signature.
        assert!(verify_manifest(FIXTURE.as_bytes(), FIXTURE_SIG, UPDATE_PUBLIC_KEY).is_err());
        assert!(verify_manifest(FIXTURE.as_bytes(), "not a signature", FIXTURE_KEY.trim()).is_err());
    }

    #[test]
    fn every_published_asset_is_known() {
        let manifest = fixture();
        for (os, arch) in [("macos", "aarch64"), ("linux", "x86_64"), ("linux", "aarch64")] {
            let asset = asset_for(os, arch).unwrap();
            assert!(manifest.assets.contains_key(asset), "{}", asset);
        }
        assert_eq!(asset_for("macos", "x86_64"), None);
        assert_eq!(asset_for("windows", "x86_64"), None);
    }

    #[test]
    fn progressive_rollout() {
        let manifest = fixture();
        let before = manifest.rollout.full_at - 1;
        assert!(rollout_allows(&manifest, 0, before));
        assert!(rollout_allows(&manifest, 9, before));
        assert!(!rollout_allows(&manifest, 10, before));
        assert!(!rollout_allows(&manifest, 99, before));
        assert!(rollout_allows(&manifest, 99, manifest.rollout.full_at));
        let buckets: std::collections::HashSet<u8> =
            (0..2000).map(|index| bucket_of(&format!("id-{}", index))).collect();
        assert!(buckets.iter().all(|bucket| *bucket < 100));
        assert!(buckets.len() > 90);
        assert_eq!(bucket_of("stable"), bucket_of("stable"));
    }

    #[test]
    fn version_crashing_at_startup_is_rolled_back() {
        let pending = || Some(Pending { from: "4.11.1".into(), to: "4.12.0".into() });
        let mut state = UpdateState { pending: pending(), rejected: vec![] };
        for crashed in 0..MAX_CRASHED_BOOTS {
            assert_eq!(on_boot(&mut state, "4.12.0", crashed), BootAction::Probation);
        }
        assert_eq!(on_boot(&mut state, "4.12.0", MAX_CRASHED_BOOTS), BootAction::RollBack);
        assert_eq!(state.pending, None);
        assert_eq!(state.rejected, vec!["4.12.0".to_string()]);

        let mut other = UpdateState { pending: pending(), rejected: vec![] };
        // A process still on the previous version leaves the probation alone.
        assert_eq!(on_boot(&mut other, "4.11.1", 5), BootAction::Continue);
        assert_eq!(other.pending, pending());
        // A newer version than the one on probation supersedes it.
        assert_eq!(on_boot(&mut other, "4.13.0", 0), BootAction::Continue);
        assert_eq!(other.pending, None);
    }

    #[test]
    fn dead_processes_count_as_crashed_boots() {
        let dir = PrivateDir::create().unwrap();
        fs::create_dir_all(dir.path.join("boots")).unwrap();
        // Our own marker never counts; pid 0x7fff_fff0 does not exist.
        fs::write(boot_marker(&dir.path, "9.0.0", std::process::id()), b"").unwrap();
        fs::write(boot_marker(&dir.path, "9.0.0", 0x7fff_fff0), b"").unwrap();
        fs::write(boot_marker(&dir.path, "8.0.0", 0x7fff_fff1), b"").unwrap();
        assert_eq!(crashed_boots(&dir.path, "9.0.0"), 1);
        remove_boot_markers(&dir.path, "9.0.0");
        assert_eq!(marker_pids(&dir.path, "9.0.0"), Vec::<u32>::new());
        assert_eq!(marker_pids(&dir.path, "8.0.0"), vec![0x7fff_fff1]);
    }

    #[test]
    fn state_round_trip_and_cleanup() {
        let dir = PrivateDir::create().unwrap();
        assert_eq!(read_state(&dir.path), UpdateState::default());
        let state = UpdateState {
            pending: Some(Pending { from: "4.11.1".into(), to: "4.12.0".into() }),
            rejected: vec!["4.11.5".into()],
        };
        write_state(&dir.path, &state).unwrap();
        assert_eq!(read_state(&dir.path), state);
        write_state(&dir.path, &UpdateState::default()).unwrap();
        assert!(!state_path(&dir.path).exists());
    }

    #[test]
    fn install_keeps_the_previous_binary() {
        let dir = PrivateDir::create().unwrap();
        let exe = dir.path.join("MemoryPilot");
        let new_binary = dir.path.join("downloaded");
        fs::write(&exe, b"old").unwrap();
        fs::write(&new_binary, b"new").unwrap();
        install_binary(&exe, &new_binary).unwrap();
        assert_eq!(fs::read(&exe).unwrap(), b"new");
        assert_eq!(fs::read(old_binary(&exe)).unwrap(), b"old");
        assert!(!exe.with_extension("new").exists());
    }

    #[test]
    fn database_backup_is_a_readable_copy_and_old_ones_are_pruned() {
        let data = PrivateDir::create().unwrap();
        let live = data.path.join("memory.db");
        let conn = rusqlite::Connection::open(&live).unwrap();
        conn.execute_batch("CREATE TABLE memories (id TEXT); INSERT INTO memories VALUES ('a'), ('b');")
            .unwrap();
        let backup = backup_database(&data.path, "9.0.0").unwrap().unwrap();
        let copy = rusqlite::Connection::open(&backup).unwrap();
        let rows: i64 = copy.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0)).unwrap();
        assert_eq!(rows, 2);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&backup).unwrap().permissions().mode() & 0o777, 0o600);
        }
        for version in ["9.0.1", "9.0.2"] {
            std::thread::sleep(Duration::from_millis(20));
            backup_database(&data.path, version).unwrap();
        }
        prune_backups(&data.path.join("backups"), BACKUPS_KEPT);
        let mut left: Vec<String> = fs::read_dir(data.path.join("backups"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, vec!["memory-before-v9.0.1.db", "memory-before-v9.0.2.db"]);
        assert_eq!(backup_database(&data.path.join("absent"), "9.0.0").unwrap(), None);
    }

    #[test]
    fn installs_that_must_not_replace_themselves() {
        let local = Path::new("/Users/someone/.local/bin/MemoryPilot");
        assert_eq!(installation_blocked(local, true), Some("debug build"));
        assert!(installation_blocked(Path::new("/repo/target/release/MemoryPilot"), false).is_some());
        assert!(installation_blocked(Path::new("/opt/homebrew/Cellar/memorypilot/4.11.0/bin/MemoryPilot"), false)
            .is_some_and(|reason| reason.contains("Homebrew")));
        if asset_for(std::env::consts::OS, std::env::consts::ARCH).is_some() {
            assert_eq!(installation_blocked(local, false), None);
        }
    }

    #[test]
    fn auto_update_switch() {
        assert!(!auto_update_disabled(None));
        assert!(!auto_update_disabled(Some("1")));
        for off in ["0", "off", "false", "no", " OFF "] {
            assert!(auto_update_disabled(Some(off)), "{}", off);
        }
    }
}
