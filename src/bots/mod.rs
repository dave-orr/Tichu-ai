//! Bundled bots. Register new ones in `by_name` to make them available to
//! the `tichu` CLI.

pub mod calls;
pub mod endgame;
pub mod h2;
pub mod heuristic;
pub mod net;
pub mod passing;
pub mod pimc;
pub mod random;

use crate::bot::Bot;

pub use calls::{CallPolicy, CallsBot};
pub use endgame::EndgameBot;
pub use h2::H2Bot;
pub use heuristic::HeuristicBot;
pub use pimc::PimcBot;
pub use random::RandomBot;

/// Names accepted by `by_name`.
pub const NAMES: &[(&str, &str)] = &[
    ("random", "uniformly random legal moves (some Tichu calls and bombs, to exercise the rules)"),
    ("heuristic", "simple rule-based baseline: dump low combos, don't overtake partner, save the Dragon"),
    ("h2", "stronger rule-based player: card tracking (bosses), hand structure, lead planning, Tichu/1-2 defense, v2 passing (Dog to an opponent)"),
    ("pimcN[@x][/F]", "search bot: N sampled deals x heuristic rollouts per candidate play; @x = rollout noise; flags F: h = h2 rollouts, i = hand inference, b = search bombs, s = search wish and Dragon gift (pimc20, pimc20@0.1, pimc10/hibs); passes with the v2 rules"),
    ("oracleN[@x]", "CHEATS (diagnostic): pimc that searches the real hands (oracle1, oracle20@0.1)"),
    ("eg:BOT", "BOT, but two-player endgames are solved exactly (e.g. eg:heuristic, eg:pimc20)"),
    ("calls:P:BOT", "BOT with call policy P: none, v1 (old heuristic rules), T = v2 Tichu threshold T + v2 Grand, sT = v2 Tichu only, g = v2 Grand only (calls:1:pimc20)"),
    ("nocalls:BOT", "same as calls:none:BOT"),
    ("pass2[~F]:BOT", "BOT with v2 passing (F = only these rules: o opponents, m Mah Jong vs Grand, p partner; e.g. pass2~o:h2): low singletons odd left / even right, Mah Jong before a Grand, partner gets my best card unless a gift would make me a Tichu hand"),
    ("net:PATH", "learned policy (a .tnet model from train/) plays its top choice; h2 for calls, passes, bombs out of turn"),
    ("vpimcN:PATH", "pimcN whose rollouts stop at my next turn and are scored by the model's value head"),
];

/// Construct a bundled bot by name.
pub fn by_name(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(RandomBot::new(seed))),
        "heuristic" => Some(Box::new(HeuristicBot::new())),
        "h2" => Some(Box::new(H2Bot::new())),
        _ if name.starts_with("h2~") => Some(Box::new(H2Bot::with_params(h2::Params::parse(&name[3..])?))),
        _ if name.starts_with("nocalls:") => Some(Box::new(CallsBot::new(CallPolicy::Never, by_name(&name[8..], seed)?))),
        _ if name.starts_with("pass2") => {
            let (head, inner) = name.split_once(':')?;
            let flags = match head.strip_prefix("pass2")? {
                "" => "",
                f => f.strip_prefix('~')?,
            };
            let cfg = passing::PassConfig::parse(flags)?;
            Some(Box::new(passing::PassBot::new(cfg, by_name(inner, seed)?)))
        }
        _ if name.starts_with("calls:") => {
            let (policy, inner) = name[6..].split_once(':')?;
            Some(Box::new(CallsBot::new(CallPolicy::parse(policy)?, by_name(inner, seed)?)))
        }
        _ if name.starts_with("net:") || name.starts_with("vpimc") => net::by_name(name, seed),
        _ if name.starts_with("eg:") => Some(Box::new(EndgameBot::new(by_name(&name[3..], seed)?))),
        _ => {
            // pimcN / oracleN, optionally @noise, optionally /flags.
            let (name, flags) = name.split_once('/').unwrap_or((name, ""));
            let (head, noise) = match name.split_once('@') {
                Some((h, x)) => (h, x.parse::<f64>().ok().filter(|x| (0.0..=1.0).contains(x))?),
                None => (name, 0.0),
            };
            let bot = if let Some(n) = head.strip_prefix("oracle") {
                PimcBot::oracle(if n.is_empty() { 1 } else { n.parse().ok()? }, seed)
            } else {
                PimcBot::new(head.strip_prefix("pimc")?.parse().ok()?, seed)
            };
            let mut bot = bot.with_noise(noise);
            let mut chars = flags.chars().peekable();
            while let Some(f) = chars.next() {
                match f {
                    'h' => bot.h2_rollouts = true,
                    'i' => {
                        // Optional digit: eps_play = 0.d (i7 = 0.7).
                        let mut cfg = crate::infer::InferConfig::default();
                        if let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                            chars.next();
                            cfg.eps_play = d as f64 / 10.0;
                        }
                        bot.infer = Some(cfg);
                    }
                    'b' => bot.search_bombs = true,
                    's' => bot.search_specials = true,
                    _ => return None,
                }
            }
            Some(Box::new(bot))
        }
    }
}
