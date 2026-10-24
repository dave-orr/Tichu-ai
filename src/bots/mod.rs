//! Bundled bots. Register new ones in `by_name` to make them available to
//! the `tichu` CLI.

pub mod calls;
pub mod endgame;
pub mod heuristic;
pub mod pimc;
pub mod random;

use crate::bot::Bot;

pub use calls::{CallPolicy, CallsBot};
pub use endgame::EndgameBot;
pub use heuristic::HeuristicBot;
pub use pimc::PimcBot;
pub use random::RandomBot;

/// Names accepted by `by_name`.
pub const NAMES: &[(&str, &str)] = &[
    ("random", "uniformly random legal moves (some Tichu calls and bombs, to exercise the rules)"),
    ("heuristic", "simple rule-based baseline: dump low combos, don't overtake partner, save the Dragon"),
    ("pimcN[@x]", "example search bot: N sampled deals x heuristic rollouts per candidate play; @x = rollout noise (pimc20, pimc20@0.1)"),
    ("oracleN[@x]", "CHEATS (diagnostic): pimc that searches the real hands (oracle1, oracle20@0.1)"),
    ("eg:BOT", "BOT, but two-player endgames are solved exactly (e.g. eg:heuristic, eg:pimc20)"),
    ("calls:P:BOT", "BOT with call policy P: none, v1 (old heuristic rules), T = v2 Tichu threshold T + v2 Grand, sT = v2 Tichu only, g = v2 Grand only (calls:1:pimc20)"),
    ("nocalls:BOT", "same as calls:none:BOT"),
];

/// Construct a bundled bot by name.
pub fn by_name(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(RandomBot::new(seed))),
        "heuristic" => Some(Box::new(HeuristicBot::new())),
        _ if name.starts_with("nocalls:") => Some(Box::new(CallsBot::new(CallPolicy::Never, by_name(&name[8..], seed)?))),
        _ if name.starts_with("calls:") => {
            let (policy, inner) = name[6..].split_once(':')?;
            Some(Box::new(CallsBot::new(CallPolicy::parse(policy)?, by_name(inner, seed)?)))
        }
        _ if name.starts_with("eg:") => Some(Box::new(EndgameBot::new(by_name(&name[3..], seed)?))),
        _ => {
            // pimcN / oracleN, optionally @noise.
            let (head, noise) = match name.split_once('@') {
                Some((h, x)) => (h, x.parse::<f64>().ok().filter(|x| (0.0..=1.0).contains(x))?),
                None => (name, 0.0),
            };
            let bot = if let Some(n) = head.strip_prefix("oracle") {
                PimcBot::oracle(if n.is_empty() { 1 } else { n.parse().ok()? }, seed)
            } else {
                PimcBot::new(head.strip_prefix("pimc")?.parse().ok()?, seed)
            };
            Some(Box::new(bot.with_noise(noise)))
        }
    }
}
