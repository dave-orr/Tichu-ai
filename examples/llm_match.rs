//! Two Claude seats (one team) against a bot, over duplicate deals, through
//! headless Claude Code on a Claude subscription. Resumable: rerun the same
//! command to pick up where it stopped. See docs/LLM_MATCH.md.
//!
//!     cargo run --release --example llm_match -- --deals 40 --opponent pimc20
//!     cargo run --release --example llm_match -- --report
//!     cargo run --release --example llm_match -- --show-prompt
//!     cargo run --release --example llm_match -- --backend stub --deals 3 --dir runs/stub

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tichu_engine::bots;
use tichu_engine::llm::client::{AskError, Reply};
use tichu_engine::llm::replay::{self, ReplayOptions};
use tichu_engine::llm::run::{self, MatchConfig, RunOptions};
use tichu_engine::llm::{Backend, ClaudeCli, Client, Decisions, OnLimit, Stub};

const USAGE: &str = "\
llm_match [options]
  --dir DIR            match directory (default runs/llm-vs-OPPONENT-sSEED)
  --deals N            total deals wanted, finished ones included (default 20)
  --opponent BOT       the bot team (default pimc20)
  --fallback BOT       makes the Claude seats' Tichu calls and covers bad answers;
                       also the paired baseline (default h2)
  --seed S             deal seed, same deals as `tichu arena --seed S` (default 1)
  --jobs N             deals played at once (default 1)
  --model M            claude --model (default opus)
  --effort E           claude --effort (low|medium|high|xhigh|max; default: CLI default)
  --claude-bin PATH    claude executable (default claude)
  --claude-arg ARG     extra argument for claude (repeatable)
  --timeout-s N        per-call timeout (default 600)
  --on-limit wait|exit on a usage limit, sleep until it resets (default) or exit with code 75
  --limit-poll-min N   wait this long when the limit message has no reset time (default 20)
  --decisions LIST     decisions Claude makes: all, or a subset of play,bomb,pass,wish,dragon
  --no-baseline        don't also play each deal with the fallback bot
  --backend claude|stub  stub = fake model, for testing the setup (no calls)
  --keep-api-key       pass ANTHROPIC_API_KEY through to claude (bills the API!)
  --report             print the report for --dir and exit
  --replay N           narrate deal N's games from --dir (both seatings) and exit;
                       add --full for Claude's whole replies, --hands for all hands
                       before each play, --baseline for the fallback bot's games
  --show-prompt        print the system prompt and sample questions, then exit (no calls)";

struct Args(Vec<String>);

impl Args {
    fn val(&self, f: &str) -> Option<String> {
        let i = self.0.iter().position(|a| a == f)?;
        match self.0.get(i + 1) {
            Some(v) => Some(v.clone()),
            None => die(&format!("{f} needs a value")),
        }
    }
    fn all(&self, f: &str) -> Vec<String> {
        self.0.windows(2).filter(|w| w[0] == f).map(|w| w[1].clone()).collect()
    }
    fn has(&self, f: &str) -> bool {
        self.0.iter().any(|a| a == f)
    }
    fn num<T: std::str::FromStr>(&self, f: &str, default: T) -> T {
        self.val(f).map(|v| v.parse().unwrap_or_else(|_| die(&format!("bad value for {f}")))).unwrap_or(default)
    }
}

fn die(msg: &str) -> ! {
    eprintln!("{msg}\n\n{USAGE}");
    std::process::exit(2)
}

/// Prints the first question of each kind (pass, lead, follow, ...), answering
/// with the stub, and exits once it has shown a play that follows a trick.
struct ShowPrompt {
    stub: Stub,
    shown: std::sync::Mutex<Vec<&'static str>>,
}

impl Backend for ShowPrompt {
    fn id(&self) -> String {
        "show-prompt".into()
    }
    fn ask(&self, system: &str, prompt: &str) -> Result<Reply, AskError> {
        let mut shown = self.shown.lock().unwrap();
        if shown.is_empty() {
            println!("===== SYSTEM PROMPT =====\n{system}\n");
        }
        let kind = if prompt.contains("Pass one card to each") {
            "pass"
        } else if prompt.contains("turn to lead") {
            "lead"
        } else if prompt.contains("It's your turn.") {
            "follow"
        } else {
            "other"
        };
        if !shown.contains(&kind) {
            shown.push(kind);
            println!("===== QUESTION ({kind}) =====\n{prompt}\n");
            if kind == "follow" {
                std::process::exit(0);
            }
        }
        self.stub.ask(system, prompt)
    }
}

fn main() {
    let args = Args(std::env::args().skip(1).collect());
    if args.has("--help") || args.has("-h") {
        println!("{USAGE}");
        return;
    }
    let opponent = args.val("--opponent").unwrap_or("pimc20".into());
    let fallback = args.val("--fallback").unwrap_or("h2".into());
    for b in [&opponent, &fallback] {
        if bots::by_name(b, 0).is_none() {
            die(&format!("unknown bot {b}"));
        }
    }
    let seed: u64 = args.num("--seed", 1);
    let decisions = Decisions::parse(&args.val("--decisions").unwrap_or("all".into()))
        .unwrap_or_else(|| die("bad --decisions"));
    let dir = PathBuf::from(args.val("--dir").unwrap_or(format!("runs/llm-vs-{}-s{seed}", opponent.replace(':', "_"))));

    let backend: Box<dyn Backend> = if args.has("--show-prompt") {
        Box::new(ShowPrompt { stub: Stub::new(0), shown: Default::default() })
    } else {
        match args.val("--backend").as_deref().unwrap_or("claude") {
            "stub" => Box::new(Stub::new(0)),
            "claude" => {
                let workdir = std::env::temp_dir().join("tichu-llm-empty");
                let mut c = ClaudeCli::new(&args.val("--model").unwrap_or("opus".into()), workdir);
                c.bin = args.val("--claude-bin").unwrap_or(c.bin);
                if c.bin.contains('/') {
                    // claude runs in an empty directory, so make the path absolute.
                    c.bin = std::fs::canonicalize(&c.bin)
                        .unwrap_or_else(|e| die(&format!("{}: {e}", c.bin)))
                        .to_string_lossy()
                        .into_owned();
                }
                c.effort = args.val("--effort");
                c.extra_args = args.all("--claude-arg");
                c.timeout = Duration::from_secs(args.num("--timeout-s", 600));
                c.keep_api_key = args.has("--keep-api-key");
                if !c.keep_api_key && std::env::var_os("ANTHROPIC_API_KEY").is_some() {
                    eprintln!("note: ANTHROPIC_API_KEY is set; not passing it to claude, so calls use your subscription login.");
                }
                Box::new(c)
            }
            other => die(&format!("unknown backend {other}")),
        }
    };
    let cfg = MatchConfig { opponent, fallback, seed, backend: backend.id(), decisions };

    if let Some(n) = args.val("--replay") {
        let deal: usize = n.parse().unwrap_or_else(|_| die("bad --replay"));
        let path = run::game_path(&dir, deal);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| die(&format!("{}: {e}", path.display())));
        let games: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| die(&e.to_string()));
        let opts = ReplayOptions { full: args.has("--full"), hands: args.has("--hands") };
        let which = if args.has("--baseline") { "baseline" } else { "llm" };
        for g in games[which].as_array().into_iter().flatten() {
            match replay::narrate(g, &opts) {
                Ok(t) => println!("{t}"),
                Err(e) => eprintln!("{e}"),
            }
        }
        return;
    }

    if args.has("--report") {
        print!("{}", run::report(&cfg, &run::read_results(&dir)));
        return;
    }

    let client = if args.has("--show-prompt") {
        Client::in_memory(backend)
    } else {
        Client::with_dir(backend, &dir).unwrap_or_else(|e| die(&format!("{}: {e}", dir.display())))
    };
    let mut client = client;
    client.on_limit = match args.val("--on-limit").as_deref().unwrap_or("wait") {
        "wait" => OnLimit::Wait,
        "exit" => OnLimit::Exit,
        _ => die("--on-limit is wait or exit"),
    };
    client.limit_poll = Duration::from_secs(60 * args.num::<u64>("--limit-poll-min", 20));
    let client = Arc::new(client);

    if args.has("--show-prompt") {
        let _ = run::play_deal(&cfg, &client, 0, false);
        return;
    }

    let opts = RunOptions {
        dir: dir.clone(),
        deals: args.num("--deals", 20),
        jobs: args.num("--jobs", 1),
        baseline: !args.has("--no-baseline"),
    };
    eprintln!("match directory: {}", dir.display());
    let result = run::run(&cfg, client.clone(), &opts);
    let st = client.stats();
    eprintln!(
        "this session: {} fresh calls, {} cached, {} failed attempts, {} usage-limit hits, ~${:.2} at API prices (not billed on a subscription)",
        st.fresh, st.cached, st.failures, st.limit_hits, st.cost_usd
    );
    match result {
        Ok(rs) => print!("\n{}", run::report(&cfg, &rs)),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
