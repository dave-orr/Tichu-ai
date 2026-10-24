//! How to write a bot and evaluate it, without touching the library.
//!
//!     cargo run --release --example custom_bot
//!
//! `LowballBot` always plays its lowest legal combination and never overtakes
//! its partner. It then plays 1000 duplicate deals against the bundled
//! heuristic bot.

use tichu_engine::arena::{self, ArenaConfig};
use tichu_engine::bot::Bot;
use tichu_engine::bots::HeuristicBot;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

struct LowballBot;

impl Bot for LowballBot {
    fn name(&self) -> String {
        "lowball".into()
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        // view.hand is sorted low to high: lowest two to the opponents, highest to partner.
        let h = &view.hand;
        PassInfo { left: h[0], right: h[1], partner: h[h.len() - 1] }
    }

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        if can_pass && view.last_played_by == Some(view.partner()) {
            return None;
        }
        // Lowest rank first; among equals, the most cards.
        (0..plays.len()).min_by(|&a, &b| {
            let (x, y) = (&plays[a], &plays[b]);
            (x.is_bomb(), x.rank, -(x.length as i32)).partial_cmp(&(y.is_bomb(), y.rank, -(y.length as i32))).unwrap()
        })
    }

    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        view.opponents()[0]
    }
}

fn main() {
    let cfg = ArenaConfig { count: 1000, ..Default::default() };
    let report = arena::run(
        "lowball",
        &|_seed| Box::new(LowballBot) as Box<dyn Bot>,
        "heuristic",
        &|_seed| Box::new(HeuristicBot::new()) as Box<dyn Bot>,
        &cfg,
    );
    print!("{}", report.summary());
}
