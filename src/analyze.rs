//! A bot's second opinion on a real round: what it would have done at each
//! decision of a logged hand (dave-orr/tichu's hand review asks for this,
//! through `tichu analyze`).
//!
//! The round is rebuilt from the server's round log (`humanlog::replay`,
//! which follows `actions` exactly), and at every decision the bot is asked
//! from the decider's own `PlayerView`, as it would be at the table: it sees
//! its hand and everything played so far, never the other hands. A fresh bot
//! answers each decision, seeded by the decision's place in the round, so the
//! same log and bot always give the same answers.
//!
//! Decisions covered: each seat's pass, every on-turn play or pass where
//! there was a choice (the server's auto-skips have none), the Mah Jong wish
//! and the Dragon gift. Not covered: Grand and Small Tichu calls, and bombs
//! out of turn.

use serde::Serialize;

use crate::bot::Bot;
use crate::bots;
use crate::engine::{Action, GameState, Pending};
use crate::humanlog::{self, RoundLog};
use crate::types::*;

/// The bot's choice at one decision.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Suggestion {
    #[serde(rename_all = "camelCase")]
    Play {
        cards: Vec<Card>,
        #[serde(skip_serializing_if = "Option::is_none")]
        phoenix_as: Option<u8>,
    },
    Pass,
    Wish { rank: Option<Rank> },
    Dragon { to: Seat },
}

/// One decision of the logged round and what the bot would have done.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Decision {
    /// Index into the log's `actions`: the action the player actually took.
    pub action: usize,
    pub seat: Seat,
    pub suggestion: Suggestion,
}

/// What a seat would have passed (cards to the left, partner and right).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PassSuggestion {
    pub seat: Seat,
    pub left: Card,
    pub partner: Card,
    pub right: Card,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Analysis {
    pub bot: String,
    pub passes: Vec<PassSuggestion>,
    pub decisions: Vec<Decision>,
}

/// A question for the bot: the state before the decision, and who decides.
enum Job {
    Pass(Seat),
    Turn { action: usize, seat: Seat, state: GameState },
    Wish { action: usize, seat: Seat, state: GameState },
    Dragon { action: usize, seat: Seat, state: GameState },
}

enum Answer {
    Pass(PassSuggestion),
    Decision(Decision),
}

fn seed_for(k: usize) -> u64 {
    0x7ac4_0000 + k as u64
}

fn answer(job: &Job, k: usize, bot: &str, pass_state: &GameState) -> Option<Answer> {
    let mut b: Box<dyn Bot> = bots::by_name(bot, seed_for(k))?;
    match job {
        Job::Pass(seat) => {
            let p = b.pass_cards(&pass_state.view(*seat));
            Some(Answer::Pass(PassSuggestion { seat: *seat, left: p.left, partner: p.partner, right: p.right }))
        }
        Job::Turn { action, seat, state } => {
            let view = state.view(*seat);
            let plays = state.legal_plays(*seat);
            let can_pass = state.can_pass(*seat);
            let suggestion = match b.play(&view, &plays, can_pass) {
                Some(i) => {
                    let c = plays.get(i)?;
                    Suggestion::Play { cards: c.cards.clone(), phoenix_as: c.phoenix_as }
                }
                None if can_pass => Suggestion::Pass,
                None => return None,
            };
            Some(Answer::Decision(Decision { action: *action, seat: *seat, suggestion }))
        }
        Job::Wish { action, seat, state } => {
            let rank = b.wish(&state.view(*seat)).filter(|r| (2..=14).contains(r));
            Some(Answer::Decision(Decision { action: *action, seat: *seat, suggestion: Suggestion::Wish { rank } }))
        }
        Job::Dragon { action, seat, state } => {
            let to = b.dragon_recipient(&state.view(*seat));
            if to % 2 == *seat % 2 || to > 3 {
                return None;
            }
            Some(Answer::Decision(Decision { action: *action, seat: *seat, suggestion: Suggestion::Dragon { to } }))
        }
    }
}

/// Ask `bot` about every decision of the logged round, on up to `threads`
/// threads. Needs a log with `actions` (dave-orr/tichu#127 or later).
pub fn analyze(log: &RoundLog, bot: &str, threads: usize) -> Result<Analysis, String> {
    if bots::by_name(bot, 0).is_none() {
        return Err(format!("unknown bot '{bot}'"));
    }
    if log.actions.is_empty() {
        return Err("the log has no actions".into());
    }
    let pass_state = humanlog::pass_state(log)?;
    let replay = humanlog::replay(log)?;

    let mut jobs: Vec<Job> = (0..4).map(Job::Pass).collect();
    // Every step but the trick awards is one logged action, in order.
    let mut action = 0usize;
    for step in &replay.steps {
        if matches!(step.action, Action::ResolveTrick) {
            continue;
        }
        let state = &step.state;
        match (&step.action, state.pending()) {
            (Action::Play { seat, .. } | Action::PassTurn { seat }, Pending::Turn(t)) if *seat == t => {
                if !state.legal_plays(t).is_empty() {
                    jobs.push(Job::Turn { action, seat: t, state: state.clone() });
                }
            }
            (Action::Wish { seat, .. }, _) => jobs.push(Job::Wish { action, seat: *seat, state: state.clone() }),
            (Action::GiveDragon { seat, .. }, _) => jobs.push(Job::Dragon { action, seat: *seat, state: state.clone() }),
            _ => {}
        }
        action += 1;
    }

    // Spread the jobs over the threads, round robin (the expensive turns are
    // spread through the hand), and put the answers back in order.
    let threads = threads.max(1).min(jobs.len().max(1));
    let mut answers: Vec<Option<Answer>> = (0..jobs.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let jobs = &jobs;
                let pass_state = &pass_state;
                scope.spawn(move || {
                    (t..jobs.len()).step_by(threads).map(|k| (k, answer(&jobs[k], k, bot, pass_state))).collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (k, a) in h.join().expect("analysis thread panicked") {
                answers[k] = a;
            }
        }
    });

    let mut out = Analysis { bot: bot.to_string(), passes: vec![], decisions: vec![] };
    for a in answers.into_iter().flatten() {
        match a {
            Answer::Pass(p) => out.passes.push(p),
            Answer::Decision(d) => out.decisions.push(d),
        }
    }
    Ok(out)
}
