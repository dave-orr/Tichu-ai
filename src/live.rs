//! `tichu live`: play on the real server (dave-orr/tichu) through its AI
//! player API (`docs/AI_API.md` there).
//!
//! The server sends each AI seat its `ClientGameState` plus the round's
//! public action log (`roundActions`). Bots need a `PlayerView` whose `log`
//! is exactly what the engine would have produced, so each seat keeps a
//! *shadow* `GameState`: the real round replayed through our engine, with the
//! hidden hands filled in by a guess. Whenever another seat plays cards the
//! guess doesn't give them, those cards are swapped in from another hidden
//! hand first. Hidden cards are interchangeable from this seat's point of
//! view (bar the ones it passed, which stay where it sent them), so the
//! shadow's view of this seat is the real one, and the engine checks every
//! step. Only `view(seat)` of the shadow ever reaches the bot.
//!
//! One `tichu live` process serves any number of seats: it polls the
//! server's matchmaking endpoint and plays each seat it gets on its own thread.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::bot::Bot;
use crate::deck::sort_hand;
use crate::engine::{Action, GameState, Pending};
use crate::humanlog::LogAction;
use crate::types::*;
use crate::view::PlayerView;

// ---------------------------------------------------------------------------
// The server's state, as much of it as we read
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerPlayer {
    #[serde(default)]
    pub name: String,
    pub tichu_call: TichuCall,
    pub has_played_first_card: bool,
    pub out_order: u8,
    pub grand_tichu_decided: bool,
    pub passed_cards: bool,
    pub card_count: usize,
    /// A bot's seat (ours or anyone's).
    #[serde(default)]
    pub is_ai: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerTeam {
    pub score: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerSettings {
    pub clockwise: bool,
    pub target_score: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerCountdown {
    pub winner: Seat,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerReceived {
    pub card: Card,
    pub from_seat: Seat,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerState {
    pub phase: String,
    pub players: Vec<ServerPlayer>,
    pub teams: Vec<ServerTeam>,
    pub current_trick: Option<Value>,
    pub turn_index: Seat,
    pub last_played_by: Option<Seat>,
    pub mah_jong_wish: Option<Rank>,
    pub mah_jong_wish_pending: bool,
    pub dragon_giveaway: bool,
    pub dragon_giveaway_by: Option<Seat>,
    pub trick_countdown: Option<ServerCountdown>,
    #[serde(default)]
    pub concede_offer: Option<Value>,
    pub settings: ServerSettings,
    pub round_number: u32,
    pub round_end_ready: Vec<Seat>,
    pub my_hand: Vec<Card>,
    pub my_seat: Seat,
    pub my_received_cards: Vec<ServerReceived>,
    /// Added by dave-orr/tichu#148; older servers don't send it.
    #[serde(default)]
    pub round_actions: Option<Vec<LogAction>>,
}

impl ServerState {
    fn actions(&self) -> &[LogAction] {
        self.round_actions.as_deref().unwrap_or(&[])
    }
    fn playing(&self) -> bool {
        self.phase == "playing"
    }
}

// ---------------------------------------------------------------------------
// The shadow game
// ---------------------------------------------------------------------------

/// One seat's replay of the real round (see the module docs).
pub struct Shadow {
    pub seat: Seat,
    g: Option<GameState>,
    /// Server round number of `g`.
    round: u32,
    /// How many of the round's `roundActions` are applied to `g`.
    applied: usize,
    /// What this seat passed (it knows where those cards went).
    my_pass: Option<PassInfo>,
    /// The round was conceded: the server ended it, the shadow stops.
    conceded: bool,
    rng: StdRng,
}

fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::GrandTichuWindow => "grandTichuWindow",
        Phase::Passing => "passing",
        Phase::Playing => "playing",
        Phase::RoundEnd => "roundEnd",
        Phase::GameEnd => "gameEnd",
    }
}

fn to_action(a: &LogAction) -> Option<Action> {
    Some(match a {
        LogAction::Play { seat, cards, phoenix_as } => Action::Play { seat: *seat, cards: cards.clone(), phoenix_as: *phoenix_as },
        LogAction::Bomb { seat, cards } => Action::Bomb { seat: *seat, cards: cards.clone() },
        LogAction::Pass { seat } => Action::PassTurn { seat: *seat },
        LogAction::SmallTichu { seat } => Action::SmallTichu { seat: *seat },
        LogAction::Wish { seat, rank } => Action::Wish { seat: *seat, rank: *rank },
        LogAction::Dragon { seat, to } => Action::GiveDragon { seat: *seat, to: *to },
        LogAction::Concede { .. } => return None,
    })
}

impl Shadow {
    pub fn new(seat: Seat, seed: u64) -> Shadow {
        Shadow { seat, g: None, round: 0, applied: 0, my_pass: None, conceded: false, rng: StdRng::seed_from_u64(seed) }
    }

    /// The shadow state, once it follows a round.
    pub fn state(&self) -> Option<&GameState> {
        self.g.as_ref()
    }

    pub fn view(&self) -> Option<PlayerView> {
        self.g.as_ref().map(|g| g.view(self.seat))
    }

    /// Note what this seat passed (after the server accepted it).
    pub fn set_my_pass(&mut self, pass: PassInfo) {
        self.my_pass = Some(pass);
        // In the log at once, as in the engine (the others' passes come when
        // the server exchanges them).
        if let Some(g) = self.g.as_mut() {
            if g.phase == Phase::Passing && g.players[self.seat].pass.is_none() {
                let _ = g.apply(&Action::PassCards { seat: self.seat, pass });
            }
        }
    }

    /// Catch up with the server. Returns the ways the shadow disagrees with
    /// the server afterwards (empty when in step), or an error when it
    /// can't follow at all (e.g. it joined mid-round).
    pub fn sync(&mut self, s: &ServerState) -> Result<Vec<String>, String> {
        if s.phase == "waiting" {
            return Ok(vec![]);
        }
        if s.phase == "grandTichuWindow" && (self.g.is_none() || self.round != s.round_number) {
            self.start_round(s);
        }
        if self.g.is_none() || self.round != s.round_number {
            return Err(format!("not following round {} (joined mid-round?)", s.round_number));
        }
        self.grand_calls(s)?;
        let g = self.g.as_mut().unwrap();
        if g.phase == Phase::Passing && s.phase == "passing" {
            // The last six cards: make my hand the real one.
            let hand = s.my_hand.clone();
            self.set_hand(&hand)?;
        }
        if self.g.as_ref().unwrap().phase == Phase::Passing && s.phase != "passing" {
            self.apply_passes(s)?;
        }
        self.apply_actions(s)?;
        let g = self.g.as_mut().unwrap();
        if !self.conceded && g.trick_countdown.is_some() && s.trick_countdown.is_none() && g.phase == Phase::Playing {
            g.apply(&Action::ResolveTrick).map_err(|e| format!("resolve trick: {e:?}"))?;
        }
        Ok(self.compare(s))
    }

    fn start_round(&mut self, s: &ServerState) {
        let settings = Settings { clockwise: s.settings.clockwise, target_score: s.settings.target_score };
        let mut g = GameState::new(settings);
        g.scores = [s.teams[0].score, s.teams[1].score];
        g.round_number = s.round_number.saturating_sub(1);
        // My first eight cards where the deal puts them, the rest at random.
        let mine = CardSet::from_cards(&s.my_hand);
        let mut rest: Vec<Card> = CardSet::FULL.minus(mine).cards();
        rest.shuffle(&mut self.rng);
        let mut deck = Vec::with_capacity(56);
        for seat in 0..4 {
            for i in 0..14 {
                if seat == self.seat && i < 8 {
                    deck.push(s.my_hand[i]);
                } else {
                    deck.push(rest.pop().unwrap());
                }
            }
        }
        g.start_round(deck);
        self.g = Some(g);
        self.round = s.round_number;
        self.applied = 0;
        self.my_pass = None;
        self.conceded = false;
    }

    fn grand_calls(&mut self, s: &ServerState) -> Result<(), String> {
        let g = self.g.as_mut().unwrap();
        if g.phase != Phase::GrandTichuWindow {
            return Ok(());
        }
        for seat in 0..4 {
            let p = &s.players[seat];
            if p.grand_tichu_decided && !g.players[seat].grand_decided {
                let call = p.tichu_call == TichuCall::Grand;
                g.apply(&Action::GrandTichu { seat, call }).map_err(|e| format!("grand call {seat}: {e:?}"))?;
            }
        }
        Ok(())
    }

    /// Cards that must stay with `seat`: the unplayed cards I passed it.
    fn pinned(&self, seat: Seat) -> CardSet {
        let g = self.g.as_ref().unwrap();
        let mut set = CardSet::EMPTY;
        if let Some(pass) = self.my_pass {
            if g.phase != Phase::Passing && g.phase != Phase::GrandTichuWindow {
                let played = CardSet::from_cards(&g.played_cards);
                for c in pass.cards() {
                    if pass.recipient(self.seat, c) == Some(seat) && !played.contains(c) {
                        set.insert(c);
                    }
                }
            }
        }
        set
    }

    /// Make `seat` (not me) hold `card`, swapping it in from another hidden
    /// hand for a card of `seat`'s that isn't in `keep` or pinned.
    fn place(&mut self, seat: Seat, card: Card, keep: CardSet) -> Result<(), String> {
        let me = self.seat;
        let pinned = self.pinned(seat).union(keep);
        let g = self.g.as_mut().unwrap();
        if g.players[seat].hand.contains(&card) {
            return Ok(());
        }
        let holder = (0..4).find(|&h| h != seat && g.players[h].hand.contains(&card));
        let Some(holder) = holder.filter(|&h| h != me) else {
            return Err(format!("seat {seat} played {card:?}, which {}", if holder == Some(me) { "is in my hand" } else { "nobody can hold" }));
        };
        let holder_pinned = {
            let _ = g;
            self.pinned(holder)
        };
        if holder_pinned.contains(card) {
            return Err(format!("seat {seat} played {card:?}, which I passed to seat {holder}"));
        }
        let g = self.g.as_mut().unwrap();
        let options: Vec<Card> = g.players[seat].hand.iter().copied().filter(|c| !pinned.contains(*c)).collect();
        let Some(&give) = options.choose(&mut self.rng) else {
            return Err(format!("seat {seat}: no free card to swap for {card:?}"));
        };
        swap(g, seat, give, holder, card);
        Ok(())
    }

    /// Make my hand exactly `hand`, swapping with the hidden hands.
    fn set_hand(&mut self, hand: &[Card]) -> Result<(), String> {
        let me = self.seat;
        let want = CardSet::from_cards(hand);
        let g = self.g.as_mut().unwrap();
        let have = CardSet::from_cards(&g.players[me].hand);
        if have == want {
            return Ok(());
        }
        let mut extra: Vec<Card> = have.minus(want).cards();
        for card in want.minus(have).cards() {
            let holder = (0..4).find(|&h| h != me && g.players[h].hand.contains(&card)).ok_or(format!("my card {card:?} is nowhere"))?;
            let give = extra.pop().ok_or("my hand is too long")?;
            swap(g, me, give, holder, card);
        }
        Ok(())
    }

    /// All four passes, once the server has exchanged them.
    fn apply_passes(&mut self, s: &ServerState) -> Result<(), String> {
        let me = self.seat;
        let mine = self.my_pass.ok_or("passes were exchanged but I don't know mine")?;
        let to_me: Vec<Option<Card>> = (0..4).map(|o| s.my_received_cards.iter().find(|r| r.from_seat == o).map(|r| r.card)).collect();
        let mut keep = CardSet::EMPTY;
        for o in (0..4).filter(|&o| o != me) {
            let card = to_me[o].ok_or(format!("no card received from seat {o}"))?;
            keep.insert(card);
        }
        let backup = self.g.clone();
        for o in (0..4).filter(|&o| o != me) {
            if let Err(e) = self.place(o, to_me[o].unwrap(), keep) {
                self.g = backup;
                return Err(e);
            }
        }
        let mine_applied = self.g.as_ref().unwrap().players[me].pass.is_some();
        let mut passes = if mine_applied { vec![] } else { vec![(me, mine)] };
        for o in (0..4).filter(|&o| o != me) {
            let card = to_me[o].unwrap();
            let g = self.g.as_ref().unwrap();
            let mut others: Vec<Card> = g.players[o].hand.iter().copied().filter(|&c| c != card).collect();
            others.shuffle(&mut self.rng);
            let (a, b) = (others[0], others[1]);
            let pass = if me == left_seat(o) {
                PassInfo { left: card, partner: a, right: b }
            } else if me == partner_seat(o) {
                PassInfo { left: a, partner: card, right: b }
            } else {
                PassInfo { left: a, partner: b, right: card }
            };
            passes.push((o, pass));
        }
        let mut g = self.g.clone().unwrap();
        for (seat, pass) in passes {
            g.apply(&Action::PassCards { seat, pass }).map_err(|e| format!("pass {seat}: {e:?}"))?;
        }
        self.g = Some(g);
        // The Mah Jong is with whoever led first (or leads now).
        let leader = s.actions().iter().find_map(|a| match a {
            LogAction::Play { seat, .. } => Some(*seat),
            _ => None,
        });
        let leader = leader.unwrap_or(s.turn_index);
        if leader != me {
            self.place(leader, MAH_JONG, CardSet::EMPTY)?;
        }
        let g = self.g.as_mut().unwrap();
        g.turn = leader;
        Ok(())
    }

    fn apply_actions(&mut self, s: &ServerState) -> Result<(), String> {
        let actions = s.actions();
        while self.applied < actions.len() && !self.conceded {
            let Some(action) = to_action(&actions[self.applied]) else {
                self.conceded = true;
                break;
            };
            let g = self.g.as_mut().unwrap();
            // A countdown ends (the trick is awarded) unless a bomb lands on it.
            if g.trick_countdown.is_some() && !matches!(action, Action::Bomb { .. } | Action::SmallTichu { .. }) {
                g.apply(&Action::ResolveTrick).map_err(|e| format!("resolve trick: {e:?}"))?;
            }
            self.apply_other(&action)?;
            self.applied += 1;
        }
        Ok(())
    }

    /// Apply a logged action, fixing up the guessed hands when it's another
    /// seat's play.
    fn apply_other(&mut self, action: &Action) -> Result<(), String> {
        let me = self.seat;
        let (seat, cards) = match action {
            Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } => (*seat, cards.clone()),
            Action::PassTurn { seat } => (*seat, vec![]),
            _ => (me, vec![]),
        };
        if seat != me {
            let keep = CardSet::from_cards(&cards);
            for &c in &cards {
                self.place(seat, c, keep)?;
            }
        }
        let g = self.g.as_mut().unwrap();
        let Err(first) = g.apply(action) else { return Ok(()) };
        // The guess may give the seat a wished card it would have had to play
        // (or the cards to play one I passed it): move those cards out, then
        // try random swaps.
        if seat != me && self.g.as_ref().unwrap().wish.is_some() {
            let keep = CardSet::from_cards(&cards);
            let rank = self.g.as_ref().unwrap().wish.unwrap();
            self.strip(seat, keep, |c| c.rank() == Some(rank) && !c.is_special());
            self.strip(seat, keep, |c| c == PHOENIX);
            for _ in 0..500 {
                let mut probe = self.g.clone().unwrap();
                if probe.apply(action).is_ok() {
                    self.g = Some(probe);
                    return Ok(());
                }
                self.random_swap(seat, keep);
            }
        }
        Err(format!("{action:?} rejected: {first:?}"))
    }

    /// Swap a random free card of `seat` with one of another hidden hand.
    fn random_swap(&mut self, seat: Seat, keep: CardSet) {
        let me = self.seat;
        let mine = self.pinned(seat).union(keep);
        let others: Vec<Seat> = (0..4).filter(|&h| h != me && h != seat && !self.g.as_ref().unwrap().players[h].is_out()).collect();
        let Some(&h) = others.choose(&mut self.rng) else { return };
        let theirs = self.pinned(h);
        let g = self.g.as_ref().unwrap();
        let a: Vec<Card> = g.players[seat].hand.iter().copied().filter(|c| !mine.contains(*c)).collect();
        let b: Vec<Card> = g.players[h].hand.iter().copied().filter(|c| !theirs.contains(*c)).collect();
        if let (Some(&x), Some(&y)) = (a.choose(&mut self.rng), b.choose(&mut self.rng)) {
            swap(self.g.as_mut().unwrap(), seat, x, h, y);
        }
    }

    /// Move every free card matching `pred` out of `seat`'s guessed hand,
    /// in exchange for non-matching ones from the other hidden hands.
    fn strip(&mut self, seat: Seat, keep: CardSet, pred: impl Fn(Card) -> bool) {
        let me = self.seat;
        let pinned = self.pinned(seat).union(keep);
        let victims: Vec<Card> =
            self.g.as_ref().unwrap().players[seat].hand.iter().copied().filter(|&c| pred(c) && !pinned.contains(c)).collect();
        for card in victims {
            let mut done = false;
            for h in (0..4).filter(|&h| h != me && h != seat) {
                let hp = self.pinned(h);
                let g = self.g.as_mut().unwrap();
                if g.players[h].is_out() {
                    continue;
                }
                let give = g.players[h].hand.iter().copied().find(|&c| !pred(c) && !hp.contains(c));
                if let Some(give) = give {
                    swap(g, h, give, seat, card);
                    done = true;
                    break;
                }
            }
            if !done {
                return;
            }
        }
    }

    /// Ways the shadow disagrees with the server.
    fn compare(&self, s: &ServerState) -> Vec<String> {
        let mut out = vec![];
        let Some(g) = &self.g else { return out };
        if self.conceded {
            return out;
        }
        let phase = phase_name(g.phase);
        let server_phase = if s.phase == "gameEnd" { "roundEnd" } else { s.phase.as_str() };
        let shadow_phase = if phase == "gameEnd" { "roundEnd" } else { phase };
        if shadow_phase != server_phase {
            out.push(format!("phase {phase} vs server {}", s.phase));
            return out;
        }
        if CardSet::from_cards(&g.players[self.seat].hand) != CardSet::from_cards(&s.my_hand) {
            out.push(format!("my hand {} vs server {}", cards_str(&g.players[self.seat].hand), cards_str(&s.my_hand)));
        }
        for seat in 0..4 {
            if g.players[seat].hand.len() != s.players[seat].card_count {
                out.push(format!("seat {seat} holds {} vs server {}", g.players[seat].hand.len(), s.players[seat].card_count));
            }
            if g.players[seat].tichu != s.players[seat].tichu_call {
                out.push(format!("seat {seat} call {:?} vs server {:?}", g.players[seat].tichu, s.players[seat].tichu_call));
            }
        }
        if s.playing() {
            if g.turn != s.turn_index {
                out.push(format!("turn {} vs server {}", g.turn, s.turn_index));
            }
            if g.wish != s.mah_jong_wish || g.wish_pending != s.mah_jong_wish_pending {
                out.push(format!("wish {:?}/{} vs server {:?}/{}", g.wish, g.wish_pending, s.mah_jong_wish, s.mah_jong_wish_pending));
            }
            if g.current_trick.is_some() != s.current_trick.is_some() {
                out.push("table differs".into());
            }
            if g.trick_countdown.is_some() != s.trick_countdown.is_some() {
                out.push("trick countdown differs".into());
            }
            if g.dragon_giveaway_by != s.dragon_giveaway_by.filter(|_| s.dragon_giveaway) {
                out.push(format!("dragon gift by {:?} vs server {:?}", g.dragon_giveaway_by, s.dragon_giveaway_by));
            }
        }
        out
    }
}

/// `a` gives `give` to `b`, and `b` gives `take` to `a`.
fn swap(g: &mut GameState, a: Seat, give: Card, b: Seat, take: Card) {
    let ha = &mut g.players[a].hand;
    ha.retain(|&c| c != give);
    ha.push(take);
    sort_hand(ha);
    let hb = &mut g.players[b].hand;
    hb.retain(|&c| c != take);
    hb.push(give);
    sort_hand(hb);
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

/// What to send, and the decision point it answers (sent at most once).
#[derive(Debug, Clone)]
pub struct Decision {
    pub key: String,
    pub action: Value,
    /// Send at once (out-of-turn bombs race the trick countdown).
    pub urgent: bool,
    /// Our own pass, to note once the server takes it.
    pub pass: Option<PassInfo>,
}

fn card_json(c: Card) -> Value {
    serde_json::to_value(c).unwrap()
}

fn cards_json(cards: &[Card]) -> Value {
    Value::Array(cards.iter().map(|&c| card_json(c)).collect())
}

/// The play of `combo` as a server action.
fn play_json(combo: &Combo) -> Value {
    let mut v = json!({"type": "play-cards", "cards": cards_json(&combo.cards)});
    if let Some(r) = combo.phoenix_as {
        v["phoenixAs"] = json!(r);
    }
    v
}

/// The next thing this seat should do, if anything. `done` says whether a
/// decision point was already answered.
pub fn decide(bot: &mut dyn Bot, shadow: &Shadow, s: &ServerState, done: &dyn Fn(&str) -> bool) -> Option<Decision> {
    let me = shadow.seat;
    let r = s.round_number;
    let n = s.actions().len();
    let d = |key: String, action: Value, urgent: bool| -> Option<Decision> {
        if done(&key) { None } else { Some(Decision { key, action, urgent, pass: None }) }
    };
    match s.phase.as_str() {
        "roundEnd" | "gameEnd" => {
            if s.round_end_ready.contains(&me) {
                return None;
            }
            return d(format!("{r}:next"), json!({"type": "next-round"}), false);
        }
        "waiting" => return None,
        _ => {}
    }
    let g = shadow.state()?;
    let view = g.view(me);
    let pl = &s.players[me];
    match s.phase.as_str() {
        "grandTichuWindow" => {
            if pl.grand_tichu_decided || done(&format!("{r}:grand")) {
                return None;
            }
            // Everyone at the table calls at once on the server, so take
            // turns: wait for every person, then for the bots in lower
            // seats. Each bot then sees every call made before it.
            let waiting = (0..4).any(|p| p != me && !s.players[p].grand_tichu_decided && (!s.players[p].is_ai || p < me));
            if waiting {
                return None;
            }
            // Never call Grand over the partner's call.
            let partner = (me + 2) % 4;
            let call = s.players[partner].tichu_call == TichuCall::None && bot.grand_tichu(&view);
            d(format!("{r}:grand"), json!({"type": "call-grand-tichu", "call": call}), false)
        }
        "passing" => {
            if pl.passed_cards || done(&format!("{r}:pass")) || view.hand.len() != 14 {
                return None;
            }
            let p = bot.pass_cards(&view);
            let action = json!({"type": "pass-cards", "left": card_json(p.left), "partner": card_json(p.partner), "right": card_json(p.right)});
            Some(Decision { key: format!("{r}:pass"), action, urgent: false, pass: Some(p) })
        }
        "playing" => {
            if s.concede_offer.is_some() {
                return None;
            }
            match g.pending() {
                Pending::Wish(seat) if seat == me && s.mah_jong_wish_pending => {
                    let rank = bot.wish(&view).filter(|r| (2..=14).contains(r));
                    return d(format!("{r}:{n}:wish"), json!({"type": "mah-jong-wish", "rank": rank}), false);
                }
                Pending::DragonGiveaway(seat) if seat == me && s.dragon_giveaway => {
                    let to = bot.dragon_recipient(&view);
                    let to = if team_for_seat(to) == team_for_seat(me) { right_seat(me) } else { to };
                    return d(format!("{r}:{n}:dragon"), json!({"type": "give-dragon-trick", "to": to}), false);
                }
                Pending::TrickCountdown(_) if s.trick_countdown.is_some() => {
                    let key = format!("{r}:{n}:countdown-bomb");
                    let bombs = g.legal_bombs(me);
                    if bombs.is_empty() || done(&key) {
                        return None;
                    }
                    let i = bot.bomb(&view, &bombs)?;
                    return d(key, json!({"type": "bomb", "cards": cards_json(&bombs.get(i)?.cards)}), true);
                }
                Pending::Turn(t) if t == me && s.turn_index == me => {
                    let small_key = format!("{r}:{n}:small");
                    if g.can_call_small_tichu(me) && !done(&small_key) && bot.small_tichu(&view) {
                        return d(small_key, json!({"type": "call-small-tichu"}), false);
                    }
                    let plays = g.legal_plays(me);
                    let can_pass = g.can_pass(me);
                    let action = match bot.play(&view, &plays, can_pass) {
                        Some(i) if i < plays.len() => play_json(&plays[i]),
                        _ if can_pass => json!({"type": "pass-turn"}),
                        _ => play_json(&plays[0]),
                    };
                    return d(format!("{r}:{n}:turn"), action, false);
                }
                Pending::Turn(_) => {
                    // Out of turn: a bomb on the play that just landed?
                    let key = format!("{r}:{n}:bomb");
                    let last_by_other = matches!(s.actions().last(), Some(LogAction::Play { seat, .. } | LogAction::Bomb { seat, .. }) if *seat != me);
                    if !last_by_other || g.current_trick.is_none() || g.wish_pending || done(&key) {
                        return None;
                    }
                    let bombs = g.legal_bombs(me);
                    if bombs.is_empty() {
                        return None;
                    }
                    let i = bot.bomb(&view, &bombs)?;
                    d(key, json!({"type": "bomb", "cards": cards_json(&bombs.get(i)?.cards)}), true)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LiveConfig {
    /// e.g. http://localhost:3000 (the `/api` prefix is added).
    pub server: String,
    pub token: String,
    pub bot: String,
    pub name: String,
    /// Join this room only (once), instead of matchmaking.
    pub room: Option<String>,
    /// Pause before each move that isn't urgent, so play is watchable.
    pub delay: Duration,
    pub poll: Duration,
    /// Stop after this many games (all seats together); 0 = run forever.
    pub games: usize,
    pub max_seats: usize,
    pub seed: u64,
}

struct Api {
    base: String,
    auth: String,
    agent: ureq::Agent,
}

enum PostError {
    /// The server answered with this status and error message.
    Status(u16, String),
    Transport(String),
}

impl std::fmt::Display for PostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PostError::Status(c, m) => write!(f, "{c} {m}"),
            PostError::Transport(m) => write!(f, "{m}"),
        }
    }
}

impl Api {
    fn new(cfg: &LiveConfig) -> Api {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).build();
        Api { base: format!("{}/api", cfg.server.trim_end_matches('/')), auth: format!("Bearer {}", cfg.token), agent }
    }

    fn post(&self, path: &str, body: Value) -> Result<Value, PostError> {
        let resp = self
            .agent
            .post(&format!("{}{path}", self.base))
            .timeout(Duration::from_secs(20))
            .set("Authorization", &self.auth)
            .send_json(body);
        match resp {
            Ok(r) => r.into_json().map_err(|e| PostError::Transport(e.to_string())),
            Err(ureq::Error::Status(code, r)) => {
                let msg = r.into_json::<Value>().ok().and_then(|v| v["error"].as_str().map(String::from)).unwrap_or_default();
                Err(PostError::Status(code, msg))
            }
            Err(e) => Err(PostError::Transport(e.to_string())),
        }
    }

    /// Open the player's event stream, by its player id where the server
    /// gave one, else by seat. Err(true) = the room or seat is gone.
    fn stream(&self, room: &str, player: Option<&str>, seat: Seat) -> Result<Box<dyn std::io::Read + Send>, bool> {
        let who = match player {
            Some(p) => format!("player={p}"),
            None => format!("seat={seat}"),
        };
        let resp = self
            .agent
            .get(&format!("{}/rooms/{room}/stream?{who}", self.base))
            .set("Authorization", &self.auth)
            .call();
        match resp {
            Ok(r) => Ok(r.into_reader()),
            Err(ureq::Error::Status(404 | 403, _)) => Err(true),
            Err(_) => Err(false),
        }
    }
}

enum Event {
    State(Box<ServerState>),
    /// The room or seat is gone for good.
    Gone,
}

/// Read the player's SSE stream on its own thread, reconnecting as needed.
/// The server can move the player to another seat in the waiting room (seat
/// swaps, random partners); `player` names it wherever it sits, and without
/// one (older servers) `seat` is where it sits now.
fn spawn_stream(api: Arc<Api>, room: String, player: Option<String>, seat: Arc<AtomicUsize>, tag: String) -> Receiver<Event> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut failures = 0;
        loop {
            let reader = match api.stream(&room, player.as_deref(), seat.load(Ordering::SeqCst)) {
                Ok(r) => r,
                Err(true) => {
                    let _ = tx.send(Event::Gone);
                    return;
                }
                Err(false) => {
                    failures += 1;
                    if failures > 60 {
                        let _ = tx.send(Event::Gone);
                        return;
                    }
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
            };
            failures = 0;
            let mut event = String::new();
            let mut data = String::new();
            for line in BufReader::new(reader).lines() {
                let Ok(line) = line else { break };
                if line.is_empty() {
                    if event == "game-state" {
                        match serde_json::from_str::<ServerState>(&data) {
                            Ok(s) => {
                                if tx.send(Event::State(Box::new(s))).is_err() {
                                    return; // the player is done
                                }
                            }
                            Err(e) => eprintln!("{tag}: bad game-state: {e}"),
                        }
                    }
                    event.clear();
                    data.clear();
                } else if let Some(v) = line.strip_prefix("event:") {
                    event = v.trim().to_string();
                } else if let Some(v) = line.strip_prefix("data:") {
                    data.push_str(v.trim_start());
                }
            }
            eprintln!("{tag}: stream closed, reconnecting");
            std::thread::sleep(Duration::from_millis(500));
        }
    });
    rx
}

/// What happened to a seat's game.
#[derive(Debug, Default, Clone)]
pub struct SeatReport {
    pub finished: bool,
    pub rounds: u32,
    pub rejected: usize,
    pub divergences: usize,
}

/// Play one seat until its game ends or the room goes away.
fn play_seat(api: Arc<Api>, cfg: &LiveConfig, room: String, player: Option<String>, mut seat: Seat, seed: u64) -> SeatReport {
    let mut tag = format!("[{room}:{seat}]");
    let seat_now = Arc::new(AtomicUsize::new(seat));
    let rx = spawn_stream(api.clone(), room.clone(), player.clone(), seat_now.clone(), tag.clone());
    let player = player.as_deref();
    let mut bot = crate::bots::by_name(&cfg.bot, seed).expect("unknown bot");
    let mut shadow = Shadow::new(seat, seed);
    let mut done: std::collections::HashMap<String, usize> = Default::default();
    let mut report = SeatReport::default();
    let mut latest: Option<Box<ServerState>> = None;
    let mut last_warning = String::new();
    // Set when a newer state arrived while waiting to move: think again
    // without blocking for the next event.
    let mut recheck = false;
    loop {
        if !recheck {
            // Wait for news, then take the newest state.
            match rx.recv_timeout(Duration::from_secs(120)) {
                Ok(e) => {
                    if absorb(e, &mut latest) {
                        eprintln!("{tag}: room closed");
                        return report;
                    }
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        recheck = false;
        for e in rx.try_iter() {
            if absorb(e, &mut latest) {
                eprintln!("{tag}: room closed");
                return report;
            }
        }
        let Some(s) = latest.as_deref().cloned() else { continue };
        let s = &s;
        if s.my_seat != seat {
            // Moved: the organizer swapped seats, or random partners
            // reshuffled the table as the game started. Nothing of the
            // round can have happened yet, so start following afresh.
            eprintln!("{tag}: moved to seat {}", s.my_seat);
            seat = s.my_seat;
            seat_now.store(seat, Ordering::SeqCst);
            tag = format!("[{room}:{seat}]");
            shadow = Shadow::new(seat, seed);
            done.clear();
            last_warning.clear();
        }
        match shadow.sync(s) {
            Ok(w) if !w.is_empty() => {
                let msg = w.join("; ");
                if msg != last_warning {
                    report.divergences += 1;
                    eprintln!("{tag}: out of step with the server: {msg}");
                    last_warning = msg;
                }
            }
            Ok(_) => last_warning.clear(),
            Err(e) => {
                if e != last_warning {
                    report.divergences += 1;
                    eprintln!("{tag}: can't follow the game: {e}");
                    last_warning = e;
                }
            }
        }
        report.rounds = report.rounds.max(s.round_number);
        let is_done = |k: &str| done.get(k).copied().unwrap_or(0) >= TRIES;
        let Some(dec) = decide(&mut *bot, &shadow, s, &is_done) else { continue };
        if !dec.urgent && !cfg.delay.is_zero() {
            // Let the table settle first; if anything changes, think again.
            let until = Instant::now() + cfg.delay;
            while let Some(left) = until.checked_duration_since(Instant::now()) {
                match rx.recv_timeout(left) {
                    Ok(e) => {
                        if absorb(e, &mut latest) {
                            return report;
                        }
                        recheck = true;
                        break;
                    }
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => return report,
                }
            }
            if recheck {
                continue;
            }
        }
        let game_end = s.phase == "gameEnd";
        let key = dec.key.clone();
        let ok = send(&api, &room, player, seat, &tag, dec, &mut shadow, &mut done, &mut report);
        if game_end && ok {
            report.finished = true;
            break;
        }
        if !ok {
            // Refused moves change nothing, so no new state will come to
            // prompt a retry: look again shortly, and after the last try fall
            // back to the simplest moves.
            if done.get(&key).copied().unwrap_or(0) >= TRIES && rescue(&api, &room, player, seat, &tag, s) && game_end {
                report.finished = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(300));
            recheck = true;
        }
    }
    report
}

/// Take in a stream event. Returns true when the room is gone.
fn absorb(e: Event, latest: &mut Option<Box<ServerState>>) -> bool {
    match e {
        Event::State(s) => {
            *latest = Some(s);
            false
        }
        Event::Gone => true,
    }
}

/// An action request: for `player` when the server gave an id (it follows
/// the player across seat moves), and the seat for older servers.
fn action_body(player: Option<&str>, seat: Seat, action: Value) -> Value {
    let mut v = json!({"seat": seat, "action": action});
    if let Some(p) = player {
        v["player"] = json!(p);
    }
    v
}

/// Tries at a decision point before falling back to `rescue`.
const TRIES: usize = 2;
/// Marks a decision point the server took.
const TAKEN: usize = usize::MAX / 2;

/// Post a decision. Returns whether the server took it.
fn send(
    api: &Api,
    room: &str,
    player: Option<&str>,
    seat: Seat,
    tag: &str,
    dec: Decision,
    shadow: &mut Shadow,
    done: &mut std::collections::HashMap<String, usize>,
    report: &mut SeatReport,
) -> bool {
    let tries = done.entry(dec.key.clone()).or_insert(0);
    match api.post(&format!("/rooms/{room}/action"), action_body(player, seat, dec.action.clone())) {
        Ok(_) => {
            *tries = TAKEN;
            if let Some(p) = dec.pass {
                shadow.set_my_pass(p);
            }
            true
        }
        Err(e) => {
            report.rejected += 1;
            *tries += 1;
            eprintln!("{tag}: {} rejected ({}/{TRIES}): {e}", dec.action, *tries);
            false
        }
    }
}

/// The bot's move keeps being refused (a bridge bug): get the game moving
/// with the simplest moves the server might take.
fn rescue(api: &Api, room: &str, player: Option<&str>, seat: Seat, tag: &str, s: &ServerState) -> bool {
    let me = seat;
    let hand = &s.my_hand;
    let mut tries: Vec<Value> = vec![];
    match s.phase.as_str() {
        "grandTichuWindow" => tries.push(json!({"type": "call-grand-tichu", "call": false})),
        "passing" if hand.len() >= 3 => tries.push(json!({"type": "pass-cards", "left": card_json(hand[0]), "partner": card_json(hand[1]), "right": card_json(hand[2])})),
        "roundEnd" | "gameEnd" => tries.push(json!({"type": "next-round"})),
        "playing" => {
            if s.mah_jong_wish_pending && s.last_played_by == Some(me) {
                tries.push(json!({"type": "mah-jong-wish", "rank": null}));
            } else if s.dragon_giveaway && s.dragon_giveaway_by == Some(me) {
                tries.push(json!({"type": "give-dragon-trick", "to": right_seat(me)}));
                tries.push(json!({"type": "give-dragon-trick", "to": left_seat(me)}));
            } else if s.turn_index == me {
                tries.push(json!({"type": "pass-turn"}));
                // Singles, the Dog last (it can only lead).
                let mut singles: Vec<Card> = hand.iter().copied().filter(|&c| c != DOG).collect();
                singles.extend(hand.iter().copied().filter(|&c| c == DOG));
                for c in singles {
                    tries.push(json!({"type": "play-cards", "cards": [card_json(c)]}));
                }
            }
        }
        _ => {}
    }
    for a in tries {
        if api.post(&format!("/rooms/{room}/action"), action_body(player, seat, a.clone())).is_ok() {
            eprintln!("{tag}: rescued with {a}");
            return true;
        }
    }
    eprintln!("{tag}: no rescue move was taken");
    false
}

/// Run the matchmaking loop: claim open AI seats and play them.
///
/// Run the matchmaking loop with bot `cfg.bot` (a `bots::by_name` spec).
pub fn run(cfg: LiveConfig) -> Vec<SeatReport> {
    let api = Arc::new(Api::new(&cfg));
    let active = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(AtomicUsize::new(0));
    let (rep_tx, rep_rx) = mpsc::channel::<SeatReport>();
    let mut reports = vec![];
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut announced_idle = false;
    loop {
        reports.extend(rep_rx.try_iter());
        if cfg.games > 0 && reports.len() >= cfg.games && active.load(Ordering::SeqCst) == 0 {
            break;
        }
        let want_more = cfg.games == 0 || started.load(Ordering::SeqCst) < cfg.games;
        if want_more && active.load(Ordering::SeqCst) < cfg.max_seats {
            let joined = match &cfg.room {
                Some(code) => api.post(&format!("/rooms/{code}/join"), json!({"name": cfg.name})),
                None => api.post("/join", json!({"name": cfg.name})),
            };
            match joined {
                Ok(v) => {
                    let (Some(room), Some(seat)) = (v["roomCode"].as_str(), v["seat"].as_u64()) else {
                        eprintln!("odd join answer: {v}");
                        std::thread::sleep(cfg.poll);
                        continue;
                    };
                    announced_idle = false;
                    let (room, seat) = (room.to_string(), seat as Seat);
                    let player = v["player"].as_str().map(String::from);
                    eprintln!("[{room}:{seat}] joined as {} ({})", cfg.name, cfg.bot);
                    active.fetch_add(1, Ordering::SeqCst);
                    started.fetch_add(1, Ordering::SeqCst);
                    let (api, cfg2, active, rep_tx) = (api.clone(), cfg.clone(), active.clone(), rep_tx.clone());
                    let seed = rng.gen();
                    std::thread::spawn(move || {
                        let r = play_seat(api, &cfg2, room.clone(), player, seat, seed);
                        eprintln!("[{room}:{seat}] done: {r:?}");
                        active.fetch_sub(1, Ordering::SeqCst);
                        let _ = rep_tx.send(r);
                    });
                    continue; // there may be more open seats
                }
                Err(PostError::Status(401, _)) => {
                    eprintln!("the server refused the bot token (401): check BOT_TOKEN");
                    std::process::exit(1);
                }
                Err(PostError::Status(404, m)) if m.contains("disabled") => {
                    eprintln!("the server's AI API is off: {m}");
                    std::process::exit(1);
                }
                Err(e) => {
                    if cfg.room.is_some() && active.load(Ordering::SeqCst) == 0 && reports.is_empty() && started.load(Ordering::SeqCst) == 0 {
                        eprintln!("can't join room: {e}");
                        std::process::exit(1);
                    }
                    if !announced_idle {
                        eprintln!("waiting for open AI seats ({e})");
                        announced_idle = true;
                    }
                }
            }
        }
        if cfg.room.is_some() && started.load(Ordering::SeqCst) > 0 && active.load(Ordering::SeqCst) == 0 {
            break;
        }
        std::thread::sleep(cfg.poll);
    }
    reports.extend(rep_rx.try_iter());
    reports
}
