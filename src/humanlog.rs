//! Real games from the production server, rebuilt through the engine.
//!
//! `npm run export-rounds` in dave-orr/tichu writes one JSON line per logged
//! round. Newer rounds carry `plays`: every accepted play and bomb in order,
//! with the trick it belongs to. Passes, the trick countdown and when a Small
//! Tichu was called aren't logged, so `replay` infers them:
//!
//! - a seat whose turn it is passed if the next logged play is someone
//!   else's and isn't a bomb on the current trick;
//! - a bomb by another seat on the current trick is applied out of turn;
//! - a trick ends (countdown, then award) when the next play opens a new one;
//! - a Small Tichu is called just before the caller's first card, or just
//!   before the first player goes out if that comes sooner;
//! - wishes and Dragon gifts come from `mahJongWishes` / `dragonGiveaways`.
//!
//! The rebuilt round must finish the way the log says (finishing order and
//! card points), otherwise `replay` returns an error.

use serde::Deserialize;

use crate::engine::{Action, GameState, Pending};
use crate::types::*;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogPlayer {
    pub seat: Seat,
    pub tichu_call: TichuCall,
    pub out_order: u8,
    pub initial_hand: Vec<Card>,
    pub passed_left: Option<Card>,
    pub passed_partner: Option<Card>,
    pub passed_right: Option<Card>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LogPlay {
    pub seat: Seat,
    pub cards: Vec<Card>,
    pub trick: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogWish {
    pub seat: Seat,
    pub rank: Option<Rank>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogDragon {
    pub from_seat: Seat,
    pub to_seat: Seat,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoundLog {
    #[serde(default)]
    pub game_id: i64,
    #[serde(default)]
    pub round_number: u32,
    pub scores_before_round: [i32; 2],
    #[serde(default)]
    pub round_card_points: [i32; 2],
    #[serde(default)]
    pub out_order: Vec<Seat>,
    pub players: Vec<LogPlayer>,
    #[serde(default)]
    pub plays: Vec<LogPlay>,
    #[serde(default)]
    pub mah_jong_wishes: Vec<LogWish>,
    #[serde(default)]
    pub dragon_giveaways: Vec<LogDragon>,
}

impl RoundLog {
    pub fn id(&self) -> String {
        format!("g{}r{}", self.game_id, self.round_number)
    }
}

/// Read a JSON Lines export. Lines that don't parse are skipped.
pub fn read_rounds(text: &str) -> Vec<RoundLog> {
    text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect()
}

/// One action of the rebuilt round and the state just before it.
#[derive(Debug, Clone)]
pub struct Step {
    pub state: GameState,
    pub action: Action,
}

impl Step {
    /// An on-turn play or pass (a decision a bot's `play` would make).
    pub fn is_turn_decision(&self) -> bool {
        matches!(self.action, Action::Play { .. } | Action::PassTurn { .. })
            && matches!(self.state.pending(), Pending::Turn(_))
    }
}

/// A rebuilt round.
#[derive(Debug, Clone)]
pub struct Replay {
    pub steps: Vec<Step>,
    /// The state after the last logged action (round over unless conceded).
    pub end: GameState,
    /// The log stopped mid-round: someone conceded (the server ends the
    /// round at once). Every step up to that point is still real play.
    pub conceded: bool,
}

fn same_cards(a: &[Card], b: &[Card]) -> bool {
    CardSet::from_cards(a) == CardSet::from_cards(b)
}

/// Rebuild a round with a play log: every applied action with the state
/// before it. The log doesn't record the table's
/// direction of play, so both are tried.
pub fn replay(r: &RoundLog) -> Result<Replay, String> {
    let mut settings = Settings::default();
    let first = replay_with(r, settings);
    if first.is_ok() {
        return first;
    }
    settings.clockwise = !settings.clockwise;
    replay_with(r, settings).map_err(|e| format!("{} / {e}", first.unwrap_err()))
}

/// `replay` with the given settings.
pub fn replay_with(r: &RoundLog, settings: Settings) -> Result<Replay, String> {
    if r.plays.is_empty() {
        return Err("no play log".into());
    }
    let mut ps: Vec<&LogPlayer> = r.players.iter().collect();
    ps.sort_by_key(|p| p.seat);
    if ps.len() != 4 || ps.iter().enumerate().any(|(i, p)| p.seat != i || p.initial_hand.len() != 14) {
        return Err("bad players".into());
    }
    let mut deck = vec![];
    for p in &ps {
        deck.extend_from_slice(&p.initial_hand);
    }
    let mut g = GameState::new(settings);
    g.start_round(deck);
    g.scores = r.scores_before_round;
    let mut steps: Vec<Step> = vec![];
    for s in 0..4 {
        apply(&mut g, Action::GrandTichu { seat: s, call: ps[s].tichu_call == TichuCall::Grand }, &mut steps)?;
    }
    for s in 0..4 {
        let p = ps[s];
        let (Some(left), Some(partner), Some(right)) = (p.passed_left, p.passed_partner, p.passed_right) else {
            return Err("missing pass".into());
        };
        apply(&mut g, Action::PassCards { seat: s, pass: PassInfo { left, partner, right } }, &mut steps)?;
    }
    // Before passes are logged as steps: keep only the play phase.
    steps.clear();

    let small: Vec<bool> = ps.iter().map(|p| p.tichu_call == TichuCall::Small).collect();
    let mut wishes = r.mah_jong_wishes.iter();
    let mut dragons = r.dragon_giveaways.iter();
    let mut trick: i32 = -1; // index of the trick on the table (server numbering)
    let mut i = 0;
    let mut guard = 0;
    let mut conceded = false;
    while i < r.plays.len() || g.phase == Phase::Playing {
        guard += 1;
        if guard > 2000 {
            return Err("replay did not converge".into());
        }
        if g.phase != Phase::Playing {
            return Err(format!("round ended with {} plays left", r.plays.len() - i));
        }
        let next = r.plays.get(i);
        match g.pending() {
            Pending::Wish(seat) => {
                let rank = match wishes.next() {
                    Some(w) if w.seat == seat => w.rank,
                    Some(w) => return Err(format!("wish by {} expected {seat}", w.seat)),
                    None => None,
                };
                apply(&mut g, Action::Wish { seat, rank }, &mut steps)?;
            }
            Pending::DragonGiveaway(seat) => {
                let d = dragons.next().ok_or("missing dragon gift")?;
                if d.from_seat != seat {
                    return Err("dragon gift from the wrong seat".into());
                }
                apply(&mut g, Action::GiveDragon { seat, to: d.to_seat }, &mut steps)?;
            }
            Pending::TrickCountdown(_) => {
                match next {
                    Some(p) if p.trick == trick && is_bomb_cards(&p.cards) => {
                        small_calls(&mut g, &mut steps, &small, p.seat, &p.cards)?;
                        apply(&mut g, Action::Bomb { seat: p.seat, cards: p.cards.clone() }, &mut steps)?;
                        i += 1;
                    }
                    _ => apply(&mut g, Action::ResolveTrick, &mut steps)?,
                }
            }
            Pending::Turn(t) => {
                let Some(p) = next else {
                    // The log stops early when the round was conceded.
                    conceded = true;
                    break;
                };
                let follows = p.trick == trick && g.current_trick.is_some();
                if p.seat == t && (follows || g.current_trick.is_none()) {
                    small_calls(&mut g, &mut steps, &small, t, &p.cards)?;
                    let leading = g.current_trick.is_none();
                    let trick_of_play = if g.current_trick.is_none() { p.trick } else { trick };
                    let action = play_action(&g, t, &p.cards, &r.plays[i + 1..], trick_of_play)?;
                    apply(&mut g, action, &mut steps)?;
                    if leading {
                        trick = p.trick;
                    }
                    i += 1;
                } else if follows && p.seat != t && is_bomb_cards(&p.cards) && g.legal_bombs(p.seat).iter().any(|b| same_cards(&b.cards, &p.cards)) {
                    small_calls(&mut g, &mut steps, &small, p.seat, &p.cards)?;
                    apply(&mut g, Action::Bomb { seat: p.seat, cards: p.cards.clone() }, &mut steps)?;
                    i += 1;
                } else if g.can_pass(t) {
                    apply(&mut g, Action::PassTurn { seat: t }, &mut steps)?;
                } else {
                    return Err(format!("seat {t} must play but the log has seat {} next", p.seat));
                }
            }
            p => return Err(format!("unexpected pending {p:?}")),
        }
    }
    if conceded {
        return Ok(Replay { steps, end: g, conceded });
    }
    let res = g.last_result().ok_or("no result")?;
    if !r.out_order.is_empty() {
        let n = r.out_order.len().min(4);
        // Only the seats that actually went out are ordered in both.
        let outs = g.players.iter().filter(|p| p.is_out()).count().min(n);
        if res.out_order[..outs] != r.out_order[..outs] {
            return Err(format!("out order {:?} != logged {:?}", res.out_order, r.out_order));
        }
    }
    if r.round_card_points != [0, 0] && res.card_points != r.round_card_points {
        return Err(format!("card points {:?} != logged {:?}", res.card_points, r.round_card_points));
    }
    Ok(Replay { steps, end: g, conceded })
}

fn apply(g: &mut GameState, a: Action, steps: &mut Vec<Step>) -> Result<(), String> {
    let before = g.clone();
    g.apply(&a).map_err(|e| format!("{a:?} rejected: {e:?}"))?;
    steps.push(Step { state: before, action: a });
    Ok(())
}

/// Small Tichu calls the log implies before `seat` plays `cards`: the
/// player's own call before their first card and, if this play takes them
/// out (after which nobody may call), every call still outstanding.
fn small_calls(g: &mut GameState, steps: &mut Vec<Step>, small: &[bool], seat: Seat, cards: &[Card]) -> Result<(), String> {
    let going_out = cards.len() == g.players[seat].hand.len();
    for s in 0..4 {
        if small[s] && g.can_call_small_tichu(s) && (s == seat || going_out) {
            apply(g, Action::SmallTichu { seat: s }, steps)?;
        }
    }
    Ok(())
}

fn is_bomb_cards(cards: &[Card]) -> bool {
    crate::combinations::identify_combo(cards, None).map(|c| c.is_bomb()).unwrap_or(false)
}

/// The on-turn play of exactly these cards. The log doesn't keep the
/// Phoenix's reading where it's ambiguous (`5 5 Ph K K`): take the reading
/// the next play on the same trick beats, else the engine's default.
fn play_action(g: &GameState, seat: Seat, cards: &[Card], later: &[LogPlay], trick: i32) -> Result<Action, String> {
    let legal: Vec<Combo> = g.legal_plays(seat).into_iter().filter(|c| same_cards(&c.cards, cards)).collect();
    if legal.is_empty() {
        return Err(format!("seat {seat} play {cards:?} not legal"));
    }
    if legal.len() > 1 {
        let next = later.iter().take_while(|p| p.trick == trick).find(|p| !is_bomb_cards(&p.cards));
        if let Some(next) = next.and_then(|p| crate::combinations::identify_combo(&p.cards, None)) {
            let fits: Vec<&Combo> = legal.iter().filter(|c| crate::combinations::can_beat(c, &next)).collect();
            if fits.len() == 1 || (fits.len() > 1 && fits.len() < legal.len()) {
                let c = fits[0];
                return Ok(Action::Play { seat, cards: c.cards.clone(), phoenix_as: c.phoenix_as });
            }
        }
    }
    let mut probe = g.clone();
    if probe.play(seat, cards, None).is_ok() {
        return Ok(Action::Play { seat, cards: cards.to_vec(), phoenix_as: None });
    }
    Ok(Action::Play { seat, cards: legal[0].cards.clone(), phoenix_as: legal[0].phoenix_as })
}
