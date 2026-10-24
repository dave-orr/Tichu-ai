//! Drives a round or a game between four bots.
//!
//! The real server lets anyone bomb at any instant. Here bombs are offered at
//! fixed points instead: after every play or bomb, every other active seat
//! that holds a beating bomb is asked (in play order), and again during the
//! trick countdown, when the trick winner is asked too. Small Tichu is offered
//! to all four seats when play begins, then before each of a seat's turns.
//! Grand Tichu and passing are simultaneous: every seat decides from a view
//! taken before anyone's choice is applied.

use rand::Rng;

use crate::bot::Bot;
use crate::deck::shuffled_deck;
use crate::engine::{Action, GameError, GameState, Pending};
use crate::scoring::RoundResult;
use crate::types::*;

#[derive(Debug)]
pub enum SimError {
    /// A bot returned something the engine rejected.
    IllegalAction { seat: Seat, action: Action, error: GameError },
    /// A bot returned an out-of-range index or passed when it couldn't.
    BadChoice { seat: Seat, what: String },
    /// The round didn't finish within the step limit (engine bug?).
    TooManySteps,
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for SimError {}

/// Optional per-action callback (for logging / tracing).
pub type Observer<'a> = &'a mut dyn FnMut(&GameState, &Action);

fn apply(state: &mut GameState, action: Action, obs: &mut Option<Observer>) -> Result<(), SimError> {
    let seat = match &action {
        Action::GrandTichu { seat, .. }
        | Action::SmallTichu { seat }
        | Action::PassCards { seat, .. }
        | Action::Play { seat, .. }
        | Action::PassTurn { seat }
        | Action::Bomb { seat, .. }
        | Action::Wish { seat, .. }
        | Action::GiveDragon { seat, .. } => *seat,
        Action::ResolveTrick => 0,
    };
    if let Some(o) = obs.as_mut() {
        o(state, &action);
    }
    state.apply(&action).map_err(|error| SimError::IllegalAction { seat, action, error })
}

/// Offer bombs to the active seats in play order starting at `from` (and
/// ending with the seat before it). Returns true if someone bombed.
fn offer_bombs(
    state: &mut GameState,
    bots: &mut [Box<dyn Bot>; 4],
    from: Seat,
    obs: &mut Option<Observer>,
) -> Result<bool, SimError> {
    let mut seat = from;
    for _ in 0..4 {
        let bombs = state.legal_bombs(seat);
        if !bombs.is_empty() {
            if let Some(i) = bots[seat].bomb(&state.view(seat), &bombs) {
                let b = bombs.get(i).ok_or(SimError::BadChoice { seat, what: format!("bomb index {i}") })?;
                apply(state, Action::Bomb { seat, cards: b.cards.clone() }, obs)?;
                return Ok(true);
            }
        }
        seat = state.next_seat(seat);
    }
    Ok(false)
}

/// After a play/bomb by `by`: let the others bomb, repeatedly, until nobody does.
fn bomb_round(
    state: &mut GameState,
    bots: &mut [Box<dyn Bot>; 4],
    by: Seat,
    obs: &mut Option<Observer>,
) -> Result<(), SimError> {
    let mut last = by;
    loop {
        if state.phase != crate::types::Phase::Playing || state.wish_pending || state.current_trick.is_none() {
            return Ok(());
        }
        // The player who just played doesn't bomb their own fresh play.
        let mut seat = state.next_seat(last);
        let mut bombed = None;
        for _ in 0..3 {
            let bombs = state.legal_bombs(seat);
            if !bombs.is_empty() {
                if let Some(i) = bots[seat].bomb(&state.view(seat), &bombs) {
                    let b = bombs.get(i).ok_or(SimError::BadChoice { seat, what: format!("bomb index {i}") })?;
                    apply(state, Action::Bomb { seat, cards: b.cards.clone() }, obs)?;
                    bombed = Some(seat);
                    break;
                }
            }
            seat = state.next_seat(seat);
        }
        match bombed {
            Some(s) => last = s,
            None => return Ok(()),
        }
    }
}

/// Play the round already dealt in `state` to completion.
pub fn play_dealt_round(
    state: &mut GameState,
    bots: &mut [Box<dyn Bot>; 4],
    obs: Option<Observer>,
) -> Result<RoundResult, SimError> {
    play_from(state, bots, obs, true)
}

/// Play on from any point in a round (e.g. a state rebuilt with
/// `GameState::from_view` inside a search bot) to the end of the round.
/// With `ask_small_tichu` false, nobody is offered a Small Tichu call.
pub fn play_from(
    state: &mut GameState,
    bots: &mut [Box<dyn Bot>; 4],
    mut obs: Option<Observer>,
    ask_small_tichu: bool,
) -> Result<RoundResult, SimError> {
    let mut asked_small = !ask_small_tichu;
    for _step in 0..5000 {
        match state.pending() {
            Pending::GrandTichu(seats) => {
                let calls: Vec<(Seat, bool)> =
                    seats.iter().map(|&s| (s, bots[s].grand_tichu(&state.view(s)))).collect();
                for (seat, call) in calls {
                    apply(state, Action::GrandTichu { seat, call }, &mut obs)?;
                }
            }
            Pending::Passing(seats) => {
                let passes: Vec<(Seat, PassInfo)> =
                    seats.iter().map(|&s| (s, bots[s].pass_cards(&state.view(s)))).collect();
                for (seat, pass) in passes {
                    apply(state, Action::PassCards { seat, pass }, &mut obs)?;
                }
            }
            Pending::Turn(seat) => {
                if !asked_small {
                    asked_small = true;
                    for i in 0..4 {
                        let s = (seat + i) % 4;
                        if state.can_call_small_tichu(s) && bots[s].small_tichu(&state.view(s)) {
                            apply(state, Action::SmallTichu { seat: s }, &mut obs)?;
                        }
                    }
                } else if ask_small_tichu
                    && state.can_call_small_tichu(seat)
                    && bots[seat].small_tichu(&state.view(seat))
                {
                    apply(state, Action::SmallTichu { seat }, &mut obs)?;
                }
                let plays = state.legal_plays(seat);
                let can_pass = state.can_pass(seat);
                match bots[seat].play(&state.view(seat), &plays, can_pass) {
                    None if can_pass => apply(state, Action::PassTurn { seat }, &mut obs)?,
                    None => return Err(SimError::BadChoice { seat, what: "passed when it must play".into() }),
                    Some(i) => {
                        let c = plays.get(i).ok_or(SimError::BadChoice { seat, what: format!("play index {i}") })?;
                        let action = Action::Play { seat, cards: c.cards.clone(), phoenix_as: c.phoenix_as };
                        apply(state, action, &mut obs)?;
                        bomb_round(state, bots, seat, &mut obs)?;
                    }
                }
            }
            Pending::Wish(seat) => {
                let rank = bots[seat].wish(&state.view(seat));
                apply(state, Action::Wish { seat, rank }, &mut obs)?;
            }
            Pending::TrickCountdown(winner) => {
                // Last call for bombs, starting after the winner and ending with them.
                let from = state.next_seat(winner);
                if offer_bombs(state, bots, from, &mut obs)? {
                    let by = state.last_played_by.unwrap();
                    bomb_round(state, bots, by, &mut obs)?;
                } else {
                    apply(state, Action::ResolveTrick, &mut obs)?;
                }
            }
            Pending::DragonGiveaway(seat) => {
                let to = bots[seat].dragon_recipient(&state.view(seat));
                apply(state, Action::GiveDragon { seat, to }, &mut obs)?;
            }
            Pending::RoundOver | Pending::GameOver => {
                let result = state.last_result().unwrap().clone();
                for s in 0..4 {
                    bots[s].round_over(&state.view(s), &result);
                }
                return Ok(result);
            }
        }
    }
    Err(SimError::TooManySteps)
}

/// Deal a fresh round from `rng` and play it.
pub fn play_round<R: Rng + ?Sized>(
    state: &mut GameState,
    bots: &mut [Box<dyn Bot>; 4],
    rng: &mut R,
    obs: Option<Observer>,
) -> Result<RoundResult, SimError> {
    state.start_round(shuffled_deck(rng));
    play_dealt_round(state, bots, obs)
}

/// Play a whole game (to `settings.target_score`). Returns the final state.
pub fn play_game<R: Rng + ?Sized>(
    settings: Settings,
    bots: &mut [Box<dyn Bot>; 4],
    rng: &mut R,
) -> Result<GameState, SimError> {
    let mut state = GameState::new(settings);
    while state.phase != Phase::GameEnd {
        play_round(&mut state, bots, rng, None)?;
        if state.round_number > 200 {
            return Err(SimError::TooManySteps);
        }
    }
    Ok(state)
}
