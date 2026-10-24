//! Bundled bots. Register new ones in `by_name` to make them available to
//! the `tichu` CLI.

pub mod heuristic;
pub mod pimc;
pub mod random;

use crate::bot::Bot;

pub use heuristic::HeuristicBot;
pub use pimc::PimcBot;
pub use random::RandomBot;

/// Names accepted by `by_name`.
pub const NAMES: &[(&str, &str)] = &[
    ("random", "uniformly random legal moves (some Tichu calls and bombs, to exercise the rules)"),
    ("heuristic", "simple rule-based baseline: dump low combos, don't overtake partner, save the Dragon"),
    ("pimcN", "example search bot: N sampled deals x heuristic rollouts per candidate play (e.g. pimc20)"),
];

/// Construct a bundled bot by name.
pub fn by_name(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(RandomBot::new(seed))),
        "heuristic" => Some(Box::new(HeuristicBot::new())),
        _ => {
            let n: usize = name.strip_prefix("pimc")?.parse().ok()?;
            Some(Box::new(PimcBot::new(n.max(1), seed)))
        }
    }
}
