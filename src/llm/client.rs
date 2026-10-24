//! Getting answers from a language model: the backends (`claude -p` and a
//! stub for tests), a persistent answer cache, and usage-limit handling.
//!
//! Every fresh answer is appended to `answers.jsonl` and flushed before it is
//! used, so a run that stops for any reason (usage limit, Ctrl-C, a crash)
//! can be restarted and replays everything it already asked for free: the
//! bots are seeded, so the replayed games reach the same positions and ask
//! the same questions.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Why a backend gave no answer.
#[derive(Debug, Clone)]
pub enum AskError {
    /// The subscription's usage limit was hit. `reset` is when it lifts, if
    /// the message said.
    UsageLimit { message: String, reset: Option<SystemTime> },
    /// Anything else (timeout, network, bad output). Worth a retry.
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub text: String,
    /// What the CLI reports the call would cost at API prices. Under a Max
    /// subscription nothing is billed; it's a rough gauge of usage.
    pub cost_usd: f64,
}

pub trait Backend: Send + Sync {
    /// Identifies the model and settings; part of the answer-cache key.
    fn id(&self) -> String;
    fn ask(&self, system: &str, prompt: &str) -> Result<Reply, AskError>;
}

// ---------------------------------------------------------------------------
// claude -p
// ---------------------------------------------------------------------------

/// Headless Claude Code. Uses whatever login the CLI has (a Max
/// subscription, after `claude` → `/login`), so nothing goes through API
/// billing. `ANTHROPIC_API_KEY` is removed from the child's environment
/// unless `keep_api_key` is set, because the CLI would otherwise bill it.
pub struct ClaudeCli {
    pub bin: String,
    pub model: String,
    pub effort: Option<String>,
    pub extra_args: Vec<String>,
    pub timeout: Duration,
    /// Empty directory to run in, so no CLAUDE.md or project settings load.
    pub workdir: PathBuf,
    pub keep_api_key: bool,
}

impl ClaudeCli {
    pub fn new(model: &str, workdir: PathBuf) -> Self {
        ClaudeCli {
            bin: "claude".into(),
            model: model.into(),
            effort: None,
            extra_args: vec![],
            timeout: Duration::from_secs(600),
            workdir,
            keep_api_key: false,
        }
    }

    fn args(&self, system: &str) -> Vec<String> {
        let mut a: Vec<String> = vec![
            "-p".into(),
            "--output-format".into(),
            "json".into(),
            "--model".into(),
            self.model.clone(),
            "--system-prompt".into(),
            system.into(),
            "--tools".into(),
            "".into(),
            "--strict-mcp-config".into(),
            "--disable-slash-commands".into(),
            "--no-session-persistence".into(),
        ];
        if let Some(e) = &self.effort {
            a.push("--effort".into());
            a.push(e.clone());
        }
        a.extend(self.extra_args.iter().cloned());
        a
    }
}

#[derive(Deserialize)]
struct CliResult {
    #[serde(default)]
    is_error: bool,
    #[serde(default)]
    subtype: String,
    #[serde(default)]
    result: Option<String>,
    #[serde(default)]
    total_cost_usd: f64,
}

impl Backend for ClaudeCli {
    fn id(&self) -> String {
        format!("claude-cli:{}:{}", self.model, self.effort.as_deref().unwrap_or("default"))
    }

    fn ask(&self, system: &str, prompt: &str) -> Result<Reply, AskError> {
        std::fs::create_dir_all(&self.workdir).map_err(|e| AskError::Failed(format!("workdir: {e}")))?;
        let mut cmd = Command::new(&self.bin);
        cmd.args(self.args(system))
            .current_dir(&self.workdir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if !self.keep_api_key {
            cmd.env_remove("ANTHROPIC_API_KEY");
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                // Not a transient failure: don't quietly play the match on the fallback bot.
                eprintln!("[llm] can't run {}: {e}", self.bin);
                std::process::exit(1);
            }
        };

        let mut stdin = child.stdin.take().unwrap();
        let input = prompt.to_string();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(input.as_bytes());
        });
        let mut out_pipe = child.stdout.take().unwrap();
        let mut err_pipe = child.stderr.take().unwrap();
        let out_reader = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = out_pipe.read_to_string(&mut s);
            s
        });
        let err_reader = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = err_pipe.read_to_string(&mut s);
            s
        });

        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(s)) => break s,
                Ok(None) if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(AskError::Failed(format!("timed out after {:?}", self.timeout)));
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(e) => return Err(AskError::Failed(format!("wait: {e}"))),
            }
        };
        let _ = writer.join();
        let stdout = out_reader.join().unwrap_or_default();
        let stderr = err_reader.join().unwrap_or_default();

        match serde_json::from_str::<CliResult>(stdout.trim()) {
            Ok(r) if !r.is_error && status.success() && r.result.as_deref().is_some_and(|t| !t.trim().is_empty()) => {
                Ok(Reply { text: r.result.unwrap(), cost_usd: r.total_cost_usd })
            }
            Ok(r) => {
                let msg = format!("{} {} {}", r.subtype, r.result.unwrap_or_default(), stderr.trim());
                Err(classify(&msg))
            }
            Err(_) => Err(classify(&format!("exit {status}: {} {}", stdout.trim(), stderr.trim()))),
        }
    }
}

/// Turn a failure message into `UsageLimit` if it reads like one.
pub fn classify(message: &str) -> AskError {
    let m = message.to_lowercase();
    let limit = m.contains("usage limit")
        || m.contains("limit reached")
        || m.contains("hit your limit")
        || m.contains("rate limit")
        || (m.contains("limit") && m.contains("reset"));
    if limit {
        AskError::UsageLimit { message: message.trim().chars().take(300).collect(), reset: reset_time(message) }
    } else {
        AskError::Failed(message.trim().chars().take(500).collect())
    }
}

/// The CLI has reported limits as `...limit reached|<unix seconds>`; pick up
/// any 10-digit number after a `|` as the reset time.
fn reset_time(message: &str) -> Option<SystemTime> {
    message.split('|').skip(1).find_map(|part| {
        let digits: String = part.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
        let secs: u64 = digits.parse().ok().filter(|_| digits.len() == 10)?;
        Some(UNIX_EPOCH + Duration::from_secs(secs))
    })
}

// ---------------------------------------------------------------------------
// Stub
// ---------------------------------------------------------------------------

/// A fake model for testing: answers with a pseudo-random valid-looking
/// choice read off the request's answer range, and replies with nonsense
/// every `garbage_every` calls (0 = never) to exercise the fallback.
pub struct Stub {
    pub garbage_every: u64,
    calls: AtomicU64,
}

impl Stub {
    pub fn new(garbage_every: u64) -> Self {
        Stub { garbage_every, calls: AtomicU64::new(0) }
    }
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

/// `(lo-hi)` or `from lo-hi` on the last line of a prompt.
fn answer_range(prompt: &str) -> Option<(i64, i64)> {
    let last = prompt.trim_end().lines().last()?;
    let nums: Vec<i64> = last
        .split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse().ok())
        .collect();
    if last.contains("three different numbers") {
        return Some((1, nums[1]));
    }
    if last.contains("2-14") {
        return Some((0, 14));
    }
    if last.contains("ANSWER: 1 or ANSWER: 2") {
        return Some((1, 2));
    }
    Some((nums[0], nums[1]))
}

impl Backend for Stub {
    fn id(&self) -> String {
        "stub".into()
    }
    fn ask(&self, _system: &str, prompt: &str) -> Result<Reply, AskError> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.garbage_every > 0 && n % self.garbage_every == 0 {
            return Ok(Reply { text: "Hmm, tough one.".into(), cost_usd: 0.0 });
        }
        let h = fnv1a(prompt.as_bytes());
        let (lo, hi) = answer_range(prompt).unwrap_or((0, 0));
        let span = (hi - lo + 1).max(1) as u64;
        let text = if prompt.trim_end().ends_with("ANSWER: 3 12 1.") {
            // Three distinct cards.
            let n = hi as u64;
            let a = h % n;
            let b = (a + 1 + (h >> 8) % (n - 1)) % n;
            let c = (0..n).find(|x| *x != a && *x != b).unwrap();
            format!("Passing.\nANSWER: {} {} {}", a + 1, b + 1, c + 1)
        } else {
            let mut pick = lo + (h % span) as i64;
            if prompt.contains("2-14 (11 = J") && pick == 1 {
                pick = 0;
            }
            format!("Stub thinking.\nANSWER: {pick}")
        };
        Ok(Reply { text, cost_usd: 0.0 })
    }
}

pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Client: cache + retries + limit handling, shared by every LLM seat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnLimit {
    /// Sleep until the limit resets (or poll), then carry on.
    Wait,
    /// Exit the process (code 75). Rerun the same command to resume.
    Exit,
}

pub const EXIT_USAGE_LIMIT: i32 = 75;

#[derive(Serialize, Deserialize)]
struct CachedAnswer {
    key: String,
    reply: String,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct ClientStats {
    pub fresh: u64,
    pub cached: u64,
    pub failures: u64,
    pub limit_hits: u64,
    pub cost_usd: f64,
}

pub struct Client {
    backend: Box<dyn Backend>,
    cache: Mutex<HashMap<String, String>>,
    cache_file: Mutex<Option<File>>,
    log_file: Mutex<Option<File>>,
    /// Nobody calls the backend before this time (set on a usage limit).
    paused_until: Mutex<Option<SystemTime>>,
    pub on_limit: OnLimit,
    /// How long to wait after a limit with no reset time.
    pub limit_poll: Duration,
    /// Attempts per question on ordinary failures (with backoff).
    pub attempts: u32,
    stats: Mutex<ClientStats>,
    /// Questions in a row that got no reply at all.
    failed_in_a_row: AtomicU64,
    /// Stop the process after this many in a row (0 = never), so a broken
    /// setup (not logged in, say) doesn't play a match on the fallback bot.
    pub max_failed_in_a_row: u64,
}

impl Client {
    /// A client with no files (tests).
    pub fn in_memory(backend: Box<dyn Backend>) -> Self {
        Client {
            backend,
            cache: Mutex::new(HashMap::new()),
            cache_file: Mutex::new(None),
            log_file: Mutex::new(None),
            paused_until: Mutex::new(None),
            on_limit: OnLimit::Wait,
            limit_poll: Duration::from_secs(20 * 60),
            attempts: 3,
            stats: Mutex::new(ClientStats::default()),
            failed_in_a_row: AtomicU64::new(0),
            max_failed_in_a_row: 3,
        }
    }

    /// A client that keeps its answer cache in `dir/answers.jsonl` and logs
    /// every decision to `dir/decisions.jsonl`.
    pub fn with_dir(backend: Box<dyn Backend>, dir: &Path) -> std::io::Result<Self> {
        let c = Client::in_memory(backend);
        std::fs::create_dir_all(dir)?;
        let cache_path = dir.join("answers.jsonl");
        if cache_path.exists() {
            let mut map = c.cache.lock().unwrap();
            for line in BufReader::new(File::open(&cache_path)?).lines() {
                // A torn last line (killed mid-write) is just skipped.
                if let Ok(a) = serde_json::from_str::<CachedAnswer>(&line?) {
                    map.insert(a.key, a.reply);
                }
            }
        }
        let open = |p: PathBuf| OpenOptions::new().create(true).append(true).open(p);
        *c.cache_file.lock().unwrap() = Some(open(cache_path)?);
        *c.log_file.lock().unwrap() = Some(open(dir.join("decisions.jsonl"))?);
        Ok(c)
    }

    pub fn backend_id(&self) -> String {
        self.backend.id()
    }

    pub fn cached_answers(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    pub fn stats(&self) -> ClientStats {
        *self.stats.lock().unwrap()
    }

    fn key(&self, system: &str, prompt: &str) -> String {
        let id = self.backend.id();
        let mut bytes = Vec::with_capacity(id.len() + system.len() + prompt.len() + 2);
        bytes.extend_from_slice(id.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(system.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(prompt.as_bytes());
        // Two independent 64-bit hashes; collisions are not a practical concern.
        let mut rev = bytes.clone();
        rev.reverse();
        format!("{:016x}{:016x}", fnv1a(&bytes), fnv1a(&rev))
    }

    /// Ask, using the cache. Returns the reply text and whether it was cached,
    /// or an error after all attempts failed. Usage limits never surface as
    /// errors: the call waits them out (or exits, with `OnLimit::Exit`).
    pub fn ask(&self, system: &str, prompt: &str) -> Result<(String, bool), String> {
        let key = self.key(system, prompt);
        if let Some(r) = self.cache.lock().unwrap().get(&key) {
            self.stats.lock().unwrap().cached += 1;
            return Ok((r.clone(), true));
        }
        let mut attempt = 0;
        loop {
            self.wait_out_pause();
            match self.backend.ask(system, prompt) {
                Ok(reply) => {
                    self.failed_in_a_row.store(0, Ordering::Relaxed);
                    self.store(&key, &reply.text);
                    let mut st = self.stats.lock().unwrap();
                    st.fresh += 1;
                    st.cost_usd += reply.cost_usd;
                    return Ok((reply.text, false));
                }
                Err(AskError::UsageLimit { message, reset }) => {
                    self.stats.lock().unwrap().limit_hits += 1;
                    self.hit_limit(&message, reset);
                }
                Err(AskError::Failed(e)) => {
                    self.stats.lock().unwrap().failures += 1;
                    attempt += 1;
                    if attempt >= self.attempts {
                        let n = self.failed_in_a_row.fetch_add(1, Ordering::Relaxed) + 1;
                        if self.max_failed_in_a_row > 0 && n >= self.max_failed_in_a_row {
                            eprintln!("[llm] {n} questions in a row got no reply; last error: {e}");
                            eprintln!("[llm] stopping (check that `claude -p hi` works and is logged in). Rerun to resume.");
                            std::process::exit(1);
                        }
                        return Err(e);
                    }
                    let backoff = Duration::from_secs(10 * 3u64.pow(attempt - 1));
                    eprintln!("[llm] call failed ({e}); retrying in {}s", backoff.as_secs());
                    std::thread::sleep(backoff);
                }
            }
        }
    }

    fn store(&self, key: &str, reply: &str) {
        self.cache.lock().unwrap().insert(key.to_string(), reply.to_string());
        if let Some(f) = self.cache_file.lock().unwrap().as_mut() {
            let line = serde_json::to_string(&CachedAnswer { key: key.into(), reply: reply.into() }).unwrap();
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }

    fn hit_limit(&self, message: &str, reset: Option<SystemTime>) {
        let until = reset.map(|t| t + Duration::from_secs(120)).unwrap_or_else(|| SystemTime::now() + self.limit_poll);
        eprintln!("\n[llm] usage limit: {message}");
        if self.on_limit == OnLimit::Exit {
            eprintln!("[llm] stopping. Every answer so far is saved; rerun the same command to resume.");
            std::process::exit(EXIT_USAGE_LIMIT);
        }
        let mut p = self.paused_until.lock().unwrap();
        if p.is_none_or(|t| t < until) {
            *p = Some(until);
        }
        eprintln!(
            "[llm] pausing until {} (local clock: in {} min). Safe to Ctrl-C; rerun the same command to resume.",
            fmt_time(until),
            until.duration_since(SystemTime::now()).unwrap_or_default().as_secs() / 60
        );
    }

    fn wait_out_pause(&self) {
        loop {
            let until = *self.paused_until.lock().unwrap();
            match until.and_then(|t| t.duration_since(SystemTime::now()).ok()) {
                Some(d) => std::thread::sleep(d.min(Duration::from_secs(30))),
                None => return,
            }
        }
    }

    /// Append one decision record to `decisions.jsonl`.
    pub fn log(&self, record: &serde_json::Value) {
        if let Some(f) = self.log_file.lock().unwrap().as_mut() {
            let _ = writeln!(f, "{record}");
            let _ = f.flush();
        }
    }
}

/// UTC timestamp, `2026-10-24 20:15:00 UTC`.
pub fn fmt_time(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Civil date from days since the epoch (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_limits() {
        assert!(matches!(
            classify("Claude AI usage limit reached|1761350400"),
            AskError::UsageLimit { reset: Some(t), .. } if t == UNIX_EPOCH + Duration::from_secs(1761350400)
        ));
        assert!(matches!(classify("You've hit your limit · resets 3pm"), AskError::UsageLimit { reset: None, .. }));
        assert!(matches!(classify("5-hour limit reached ∙ resets 9pm"), AskError::UsageLimit { .. }));
        assert!(matches!(classify("API Error: 529 overloaded"), AskError::Failed(_)));
    }

    #[test]
    fn formats_time() {
        assert_eq!(fmt_time(UNIX_EPOCH + Duration::from_secs(1761350400)), "2025-10-25 00:00:00 UTC");
    }

    struct Flaky {
        n: AtomicU64,
    }
    impl Backend for Flaky {
        fn id(&self) -> String {
            "flaky".into()
        }
        fn ask(&self, _: &str, _: &str) -> Result<Reply, AskError> {
            match self.n.fetch_add(1, Ordering::Relaxed) {
                0 => Err(AskError::UsageLimit { message: "limit reached".into(), reset: None }),
                _ => Ok(Reply { text: "ANSWER: 1".into(), cost_usd: 0.01 }),
            }
        }
    }

    #[test]
    fn waits_out_a_limit_and_caches() {
        let mut c = Client::in_memory(Box::new(Flaky { n: AtomicU64::new(0) }));
        c.limit_poll = Duration::from_millis(50);
        let t = Instant::now();
        assert_eq!(c.ask("s", "p").unwrap(), ("ANSWER: 1".to_string(), false));
        assert!(t.elapsed() >= Duration::from_millis(50));
        assert_eq!(c.ask("s", "p").unwrap(), ("ANSWER: 1".to_string(), true));
        let st = c.stats();
        assert_eq!((st.fresh, st.cached, st.limit_hits), (1, 1, 1));
    }
}
