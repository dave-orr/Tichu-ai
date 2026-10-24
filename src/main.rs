use std::process::exit;

use rand::rngs::StdRng;
use rand::SeedableRng;

use tichu_engine::arena::{self, ArenaConfig, Mode};
use tichu_engine::bot::{Bot, BotFactory};
use tichu_engine::bots;
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::{Action, GameState};
use tichu_engine::sim::play_dealt_round;
use tichu_engine::trace::trace_round;
use tichu_engine::types::*;

const USAGE: &str = "\
usage:
  tichu bots                              list bundled bots
  tichu arena <A> <B> [options]           pit bot A against bot B
      --rounds N      score N independent deals by point margin (default 2000)
      --games N       play N full games to 1000 instead
      --seed S        RNG seed (default 1)
      --threads T     worker threads (default: all cores)
      --no-duplicate  play each deal once instead of once per seating
      --clockwise     clockwise play direction
      --log DIR       write game logs (deal-NNNN.json, both seatings) for
                      the first --log-deals deals (default 20) to DIR
  tichu replay <LOG.json>... [--out FILE] [--json]
                                          build the visual replay viewer
                                          (one HTML file, default replay.html)
                                          from game logs (tichu arena --log,
                                          llm_match games/); --json prints the
                                          replay data instead
  tichu play <A> <B> [--seed S]           narrate one round (A = seats 0/2)
  tichu trace [--rounds N] [--seed S] [--bot NAME]
                                          JSONL round traces for tools/ts-diff";

fn factory(name: &str) -> Box<BotFactory> {
    if bots::by_name(name, 0).is_none() {
        eprintln!("unknown bot '{name}'. Bundled bots:");
        list_bots();
        exit(2);
    }
    let name = name.to_string();
    Box::new(move |seed| bots::by_name(&name, seed).unwrap())
}

fn list_bots() {
    for (name, desc) in bots::NAMES {
        eprintln!("  {name:<12} {desc}");
    }
}

struct Args {
    positional: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(raw: &[String]) -> Args {
        let mut a = Args { positional: vec![], flags: vec![] };
        let mut i = 0;
        while i < raw.len() {
            let s = &raw[i];
            if let Some(f) = s.strip_prefix("--") {
                let takes_value = !matches!(f, "no-duplicate" | "clockwise" | "json" | "body-only");
                let v = if takes_value { i += 1; raw.get(i).cloned() } else { None };
                if takes_value && v.is_none() {
                    eprintln!("--{f} needs a value");
                    exit(2);
                }
                a.flags.push((f.to_string(), v));
            } else {
                a.positional.push(s.clone());
            }
            i += 1;
        }
        a
    }
    fn has(&self, f: &str) -> bool {
        self.flags.iter().any(|(k, _)| k == f)
    }
    fn num<T: std::str::FromStr>(&self, f: &str) -> Option<T> {
        self.flags.iter().find(|(k, _)| k == f).map(|(_, v)| {
            v.as_deref().unwrap().parse().unwrap_or_else(|_| {
                eprintln!("bad value for --{f}");
                exit(2)
            })
        })
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        eprintln!("{USAGE}");
        exit(2);
    };
    let args = Args::parse(&raw[1..]);
    let seed: u64 = args.num("seed").unwrap_or(1);
    let settings = Settings { clockwise: args.has("clockwise"), ..Settings::default() };

    match cmd.as_str() {
        "bots" => list_bots(),
        "arena" => {
            let [a, b] = two_bots(&args);
            let mut cfg = ArenaConfig { seed, settings, duplicate: !args.has("no-duplicate"), ..Default::default() };
            if let Some(t) = args.num("threads") {
                cfg.threads = t;
            }
            if let Some(n) = args.num("games") {
                cfg.mode = Mode::Games;
                cfg.count = n;
            } else {
                cfg.count = args.num("rounds").unwrap_or(2000);
            }
            if let Some((_, Some(dir))) = args.flags.iter().find(|(k, _)| k == "log") {
                cfg.log_dir = Some(dir.into());
                cfg.log_deals = args.num("log-deals").unwrap_or(20);
            }
            let start = std::time::Instant::now();
            let report = arena::run(&a, &*factory(&a), &b, &*factory(&b), &cfg);
            print!("{}", report.summary());
            println!("  ({:.1}s)", start.elapsed().as_secs_f64());
            if !report.errors.is_empty() {
                exit(1);
            }
        }
        "replay" => {
            if args.positional.is_empty() {
                eprintln!("{USAGE}");
                exit(2);
            }
            let mut replays = Vec::new();
            for path in &args.positional {
                let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
                    eprintln!("{path}: {e}");
                    exit(1)
                });
                let before = replays.len();
                match tichu_engine::replay::build_from_text(&text) {
                    Ok(r) => replays.extend(r),
                    Err(e) => {
                        eprintln!("{path}: {e}");
                        exit(1);
                    }
                }
                if replays.len() == before {
                    eprintln!("{path}: no game logs found");
                }
            }
            if args.has("json") {
                println!("{}", serde_json::Value::from(replays));
            } else {
                let out = args.flags.iter().find(|(k, _)| k == "out").and_then(|(_, v)| v.clone()).unwrap_or("replay.html".into());
                let page = if args.has("body-only") { tichu_engine::replay::html_body(&replays) } else { tichu_engine::replay::html(&replays) };
                std::fs::write(&out, page).unwrap_or_else(|e| {
                    eprintln!("{out}: {e}");
                    exit(1)
                });
                eprintln!("wrote {out} ({} game{})", replays.len(), if replays.len() == 1 { "" } else { "s" });
            }
        }
        "play" => {
            let [a, b] = two_bots(&args);
            let (fa, fb) = (factory(&a), factory(&b));
            let mut bots: [Box<dyn Bot>; 4] = [fa(seed), fb(seed + 1), fa(seed + 2), fb(seed + 3)];
            narrate(&mut bots, seed, settings);
        }
        "trace" => {
            let n: usize = args.num("rounds").unwrap_or(100);
            let name = args.flags.iter().find(|(k, _)| k == "bot").and_then(|(_, v)| v.clone());
            let name = name.unwrap_or_else(|| "random".into());
            let f = factory(&name);
            let mut rng = StdRng::seed_from_u64(seed);
            for i in 0..n {
                // Alternate play direction so both are covered.
                let settings = Settings { clockwise: i % 2 == 1, ..Settings::default() };
                let mut state = GameState::new(settings);
                let s = seed.wrapping_mul(1000).wrapping_add(i as u64 * 4);
                let mut bots: [Box<dyn Bot>; 4] = [f(s), f(s + 1), f(s + 2), f(s + 3)];
                let deck = shuffled_deck(&mut rng);
                println!("{}", trace_round(&mut state, &mut bots, deck));
            }
        }
        _ => {
            eprintln!("{USAGE}");
            exit(2);
        }
    }
}

fn two_bots(args: &Args) -> [String; 2] {
    match args.positional.as_slice() {
        [a, b] => [a.clone(), b.clone()],
        _ => {
            eprintln!("{USAGE}");
            exit(2);
        }
    }
}

fn narrate(bots: &mut [Box<dyn Bot>; 4], seed: u64, settings: Settings) {
    let names: Vec<String> = bots.iter().map(|b| b.name()).collect();
    let mut state = GameState::new(settings);
    let mut rng = StdRng::seed_from_u64(seed);
    state.start_round(shuffled_deck(&mut rng));
    let mut obs = |s: &GameState, a: &Action| {
        let who = |seat: &Seat| format!("{} (seat {seat})", names[*seat]);
        let line = match a {
            Action::GrandTichu { seat, call: true } => format!("{} calls GRAND TICHU", who(seat)),
            Action::GrandTichu { .. } => return,
            Action::PassCards { seat, pass } => format!(
                "{} passes {} left, {} partner, {} right",
                who(seat), pass.left, pass.partner, pass.right
            ),
            Action::SmallTichu { seat } => format!("{} calls TICHU", who(seat)),
            Action::Play { seat, cards, .. } => {
                let left: Vec<Card> = s.players[*seat].hand.iter().filter(|c| !cards.contains(c)).copied().collect();
                format!("{} plays {:<22} | left: {}", who(seat), cards_str(cards), cards_str(&left))
            }
            Action::PassTurn { seat } => format!("{} passes", who(seat)),
            Action::Bomb { seat, cards } => format!("{} BOMBS {}", who(seat), cards_str(cards)),
            Action::Wish { seat, rank } => format!("{} wishes for {:?}", who(seat), rank),
            Action::ResolveTrick => {
                let w = s.trick_countdown.unwrap();
                let pts: i32 = s.trick_plays.iter().map(|p| tichu_engine::scoring::total_points(&p.combo.cards)).sum();
                format!("  -> trick to {} ({pts} pts)", who(&w))
            }
            Action::GiveDragon { seat, to } => format!("{} gives the Dragon trick to {}", who(seat), who(to)),
        };
        if let Action::PassCards { seat: 0, .. } = a {
            for (i, p) in s.players.iter().enumerate() {
                println!("seat {i} ({}): {}", names[i], cards_str(&p.hand));
            }
        }
        println!("{line}");
    };
    match play_dealt_round(&mut state, bots, Some(&mut obs)) {
        Ok(r) => println!(
            "\nround over: out order {:?}, card points {:?}, tichu {:?} => team0 {:+}, team1 {:+}",
            r.out_order, r.card_points, r.tichu_bonuses, r.total(0), r.total(1)
        ),
        Err(e) => println!("\nerror: {e}"),
    }
}
