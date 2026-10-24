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
    ("pimcN[@x]", "example search bot: N sampled deals x heuristic rollouts per candidate play; @x = rollout noise (pimc20, pimc20@0.1)"),
    ("oracleN[@x]", "CHEATS (diagnostic): pimc that searches the real hands (oracle1, oracle20@0.1)"),
    ("nocalls:BOT", "BOT, but never calls Tichu or Grand Tichu (e.g. nocalls:pimc20)"),
];

/// Construct a bundled bot by name.
pub fn by_name(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(RandomBot::new(seed))),
        "heuristic" => Some(Box::new(HeuristicBot::new())),
        _ if name.starts_with("nocalls:") => Some(Box::new(NoCalls(by_name(&name[8..], seed)?))),
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

/// Wraps a bot but never calls Tichu (for measuring what calls are worth).
pub struct NoCalls(pub Box<dyn Bot>);

impl Bot for NoCalls {
    fn name(&self) -> String {
        format!("nocalls:{}", self.0.name())
    }
    fn pass_cards(&mut self, view: &crate::view::PlayerView) -> crate::types::PassInfo {
        self.0.pass_cards(view)
    }
    fn play(&mut self, view: &crate::view::PlayerView, plays: &[crate::types::Combo], can_pass: bool) -> Option<usize> {
        self.0.play(view, plays, can_pass)
    }
    fn bomb(&mut self, view: &crate::view::PlayerView, bombs: &[crate::types::Combo]) -> Option<usize> {
        self.0.bomb(view, bombs)
    }
    fn wish(&mut self, view: &crate::view::PlayerView) -> Option<crate::types::Rank> {
        self.0.wish(view)
    }
    fn dragon_recipient(&mut self, view: &crate::view::PlayerView) -> crate::types::Seat {
        self.0.dragon_recipient(view)
    }
    fn cheats(&self) -> bool {
        self.0.cheats()
    }
    fn peek(&mut self, state: &crate::engine::GameState) {
        self.0.peek(state)
    }
}
