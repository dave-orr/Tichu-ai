//! The interface a Tichu bot implements. Every method gets the bot's own
//! `PlayerView` (never the full game state) plus, where there's a choice, the
//! legal options — so a bot can't make an illegal move, and any bot that
//! implements this trait can be dropped into `sim` / `arena`.

use crate::types::*;
use crate::view::PlayerView;

pub trait Bot {
    /// Short name, used in arena reports.
    fn name(&self) -> String;

    /// After seeing the first 8 cards: call Grand Tichu?
    fn grand_tichu(&mut self, _view: &PlayerView) -> bool {
        false
    }

    /// Call Small Tichu? Asked once when play begins (after the pass), and
    /// again before each of the bot's turns while it is still allowed.
    fn small_tichu(&mut self, _view: &PlayerView) -> bool {
        false
    }

    /// Choose 3 distinct cards from `view.hand` to pass.
    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo;

    /// On my turn: return an index into `plays`, or None to pass. `None` is
    /// only legal when `can_pass` (it is false when leading or when the
    /// Mah Jong wish forces a play); `plays` is never empty then.
    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize>;

    /// Out-of-turn bomb opportunity: someone just played, or a trick is about
    /// to be awarded. Return an index into `bombs` to throw one.
    fn bomb(&mut self, _view: &PlayerView, _bombs: &[Combo]) -> Option<usize> {
        None
    }

    /// I just played the Mah Jong: wish for a rank 2..=14, or None.
    fn wish(&mut self, _view: &PlayerView) -> Option<Rank> {
        None
    }

    /// I won a trick with the Dragon: which opponent gets it?
    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat;

    /// Diagnostic oracles only: return true to have the simulator call `peek`
    /// with the full game state (every hand) before each of this bot's
    /// decisions. Used to measure how much hidden information costs; a real
    /// bot must leave this false.
    fn cheats(&self) -> bool {
        false
    }

    fn peek(&mut self, _state: &crate::engine::GameState) {}

    /// Called on all four bots when a round ends (for learning / logging).
    fn round_over(&mut self, _view: &PlayerView, _result: &crate::scoring::RoundResult) {}
}

/// Constructs a fresh bot from a seed (one per seat, per game).
pub type BotFactory = dyn Fn(u64) -> Box<dyn Bot> + Sync;
