//! Fixed-size features of a `PlayerView` and of each candidate play, for
//! learned bots (`bots::net`) and the self-play exporter (`selfplay`).
//!
//! Policy nets score each candidate separately (`state ÃƒÂ¢Ã…Â Ã¢â‚¬Â¢ action ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ logit`),
//! since the number of legal plays varies from turn to turn.
//!
//! Every feature is a byte; the value the net sees is `byte / 255`, so
//! training (Python reads the bytes from the export) and inference (Rust)
//! see exactly the same inputs. Bump `FEATURE_VERSION` whenever the layout
//! changes: model files record the version they were trained on, and the net
//! bot refuses a mismatch.
//!
//! Seats are relative, in play order: 0 = me, 1 = the next player, 2 =
//! partner, 3 = the player before me. That makes the features independent of
//! the seat number and of the play direction.

use crate::bots::h2::{self, Counts};
use crate::combinations::identify_combo;
use crate::engine::Action;
use crate::scoring::total_points;
use crate::types::*;
use crate::view::PlayerView;

pub const FEATURE_VERSION: u32 = 2;
pub const STATE_DIM: usize = 704;
pub const ACTION_DIM: usize = 48;

const ON: u8 = 255;

struct Enc<'a>(&'a mut Vec<u8>);

impl Enc<'_> {
    fn bit(&mut self, b: bool) {
        self.0.push(if b { ON } else { 0 });
    }
    /// `x / max`, clamped to [0, 1].
    fn frac(&mut self, x: f32, max: f32) {
        self.0.push(((x / max).clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    /// `(x - lo) / (hi - lo)`, clamped to [0, 1].
    fn range(&mut self, x: f32, lo: f32, hi: f32) {
        self.frac(x - lo, hi - lo)
    }
    fn set(&mut self, s: CardSet) {
        for i in 0..56 {
            self.bit(s.0 & (1u64 << i) != 0);
        }
    }
    fn one_hot(&mut self, i: Option<usize>, n: usize) {
        for k in 0..n {
            self.bit(i == Some(k));
        }
    }
}

/// `seat` relative to `view.seat`, counted in play order.
pub fn rel(view: &PlayerView, seat: Seat) -> usize {
    let mut s = view.seat;
    for k in 0..4 {
        if s == seat {
            return k;
        }
        s = view.next_seat(s);
    }
    unreachable!()
}

fn kind_index(k: ComboKind) -> usize {
    match k {
        ComboKind::Single => 0,
        ComboKind::Pair => 1,
        ComboKind::Triple => 2,
        ComboKind::FullHouse => 3,
        ComboKind::Straight => 4,
        ComboKind::ConsecutivePairs => 5,
        ComboKind::FourOfAKindBomb => 6,
        ComboKind::StraightFlushBomb => 7,
    }
}

fn rank_counts(s: CardSet) -> [u8; 15] {
    let mut n = [0u8; 15];
    for c in s.cards() {
        if let Some(r) = c.rank() {
            n[r as usize] += 1;
        }
    }
    n
}

/// Four-of-a-kinds plus maximal same-suit runs of 5+ (a cheap bomb count).
fn bomb_count(s: CardSet) -> usize {
    let counts = rank_counts(s);
    let mut n = (2..=14).filter(|&r| counts[r] == 4).count();
    for suit in Suit::ALL {
        let mut run = 0;
        for r in 2..=14u8 {
            if s.contains(Card::normal(suit, r)) {
                run += 1;
                if run == 5 {
                    n += 1;
                }
            } else {
                run = 0;
            }
        }
    }
    n
}

/// A card's rank class: 2..=14 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ 0..=12, then Mah Jong, Dog, Phoenix, Dragon.
fn rank_class(c: Card) -> usize {
    match c {
        Card::Normal { rank, .. } => rank as usize - 2,
        Card::Special(SpecialKind::MahJong) => 13,
        Card::Special(SpecialKind::Dog) => 14,
        Card::Special(SpecialKind::Phoenix) => 15,
        Card::Special(SpecialKind::Dragon) => 16,
    }
}

/// How strong a bomb is, in (0, 1]: any straight flush beats any four of a
/// kind; then by length and top rank.
fn bomb_strength(kind: ComboKind, top: f32, len: usize) -> f32 {
    match kind {
        ComboKind::FourOfAKindBomb => 0.05 + 0.03 * top,
        _ => (0.5 + 0.04 * len.saturating_sub(5) as f32 + 0.02 * top).min(1.0),
    }
}

/// My strongest bomb (0 = none).
fn best_bomb(hand: CardSet) -> f32 {
    let counts = rank_counts(hand);
    let mut best = (2..=14).filter(|&r| counts[r] == 4).map(|r| bomb_strength(ComboKind::FourOfAKindBomb, r as f32, 4)).fold(0.0, f32::max);
    for suit in Suit::ALL {
        let mut run = 0;
        for r in 2..=14u8 {
            run = if hand.contains(Card::normal(suit, r)) { run + 1 } else { 0 };
            if run >= 5 {
                best = best.max(bomb_strength(ComboKind::StraightFlushBomb, r as f32, run));
            }
        }
    }
    best
}

/// What I can infer about another seat's bombs.
#[derive(Clone, Copy, Default)]
struct Threat {
    /// Expected number of four of a kinds it holds.
    quads: f32,
    /// Chance it holds at least one bomb (treating candidate bombs as
    /// independent).
    any: f32,
    /// Strength of the strongest bomb it could hold (0 = none possible).
    best: f32,
}

/// Bomb threats by relative seat (index 0, me, is unused). Every four of a
/// kind, and every five-card straight flush, that no card I've seen rules
/// out for that seat, with the chance the seat holds it if the unknown cards
/// were dealt at random. Cards I passed count as known to be in their
/// recipient's hand.
fn bomb_threats(view: &PlayerView) -> [Threat; 4] {
    let mut out = [Threat::default(); 4];
    let seen = CardSet::from_cards(&view.hand).union(CardSet::from_cards(&view.played_cards));
    let mut known = [CardSet::EMPTY; 4];
    for (c, s) in view.known_holdings() {
        known[s].insert(c);
    }
    let all_known = known.iter().fold(CardSet::EMPTY, |a, k| a.union(*k));
    let u = view.unseen().minus(all_known);
    let n = u.len();
    let mut bombs: Vec<(CardSet, f32)> = vec![];
    for r in 2..=14u8 {
        let s = Suit::ALL.iter().fold(CardSet::EMPTY, |mut a, &suit| {
            a.insert(Card::normal(suit, r));
            a
        });
        bombs.push((s, bomb_strength(ComboKind::FourOfAKindBomb, r as f32, 4)));
    }
    for suit in Suit::ALL {
        for lo in 2..=10u8 {
            let s = (lo..lo + 5).fold(CardSet::EMPTY, |mut a, r| {
                a.insert(Card::normal(suit, r));
                a
            });
            bombs.push((s, bomb_strength(ComboKind::StraightFlushBomb, (lo + 4) as f32, 5)));
        }
    }
    for seat in 0..4 {
        let p = &view.players[seat];
        if seat == view.seat || p.is_out() {
            continue;
        }
        let m = p.card_count.saturating_sub(known[seat].len());
        let blocked = seen.union(all_known.minus(known[seat]));
        let t = &mut out[rel(view, seat)];
        let mut none = 1.0f32;
        for (i, &(cards, strength)) in bombs.iter().enumerate() {
            if !cards.intersect(blocked).is_empty() || p.card_count < cards.len() {
                continue;
            }
            let k = cards.intersect(u).len();
            let prob = (0..k).map(|j| m.saturating_sub(j) as f32 / (n - j) as f32).product::<f32>();
            if prob <= 0.0 {
                continue;
            }
            if i < 13 {
                t.quads += prob;
            }
            none *= 1.0 - prob;
            t.best = t.best.max(strength);
        }
        t.any = 1.0 - none;
    }
    out
}

/// What the action log says about each seat beyond the cards it played
/// (indexed by relative seat).
struct History {
    /// Per combo kind (single ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ consecutive pairs): the lowest rank the seat
    /// passed on while an opponent of theirs held the trick (0 = never).
    passed_on: [[f32; 6]; 4],
    /// Leads per combo kind (bombs aside), and the Dog at index 6.
    leads: [[u8; 7]; 4],
    /// Tricks completed this round.
    tricks: usize,
}

fn history(view: &PlayerView) -> History {
    let mut h = History { passed_on: [[0.0; 6]; 4], leads: [[0; 7]; 4], tricks: 0 };
    let mut cur: Option<(Combo, Seat)> = None;
    for a in &view.log {
        match a {
            Action::Play { seat, cards, phoenix_as } => {
                let r = rel(view, *seat);
                if cards.as_slice() == [DOG] {
                    h.leads[r][6] = h.leads[r][6].saturating_add(1);
                    cur = None;
                    continue;
                }
                let Some(mut c) = identify_combo(cards, *phoenix_as) else { continue };
                match &cur {
                    None => {
                        let k = kind_index(c.kind);
                        if k < 6 {
                            h.leads[r][k] = h.leads[r][k].saturating_add(1);
                        }
                    }
                    Some((t, _)) => {
                        if c.kind == ComboKind::Single && cards[0] == PHOENIX {
                            c.rank = t.rank + 0.5;
                        }
                    }
                }
                cur = Some((c, *seat));
            }
            Action::Bomb { seat, cards } => {
                cur = identify_combo(cards, None).map(|c| (c, *seat));
            }
            Action::PassTurn { seat } => {
                if let Some((t, holder)) = &cur {
                    let k = kind_index(t.kind);
                    if k < 6 && team_for_seat(*holder) != team_for_seat(*seat) {
                        let v = &mut h.passed_on[rel(view, *seat)][k];
                        if *v == 0.0 || t.rank < *v {
                            *v = t.rank;
                        }
                    }
                }
            }
            Action::ResolveTrick => {
                cur = None;
                h.tricks += 1;
            }
            _ => {}
        }
    }
    h
}

/// Encode what `view.seat` sees at one of its play decisions.
pub fn encode_state(view: &PlayerView, out: &mut Vec<u8>) {
    let start = out.len();
    let mut e = Enc(out);
    let hand = CardSet::from_cards(&view.hand);
    e.set(hand);

    // Cards played this round, by who played them.
    let mut played = [CardSet::EMPTY; 4];
    for a in &view.log {
        if let Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } = a {
            for c in cards {
                played[rel(view, *seat)].insert(*c);
            }
        }
    }
    for p in played {
        e.set(p);
    }

    // Unplayed cards I passed, by who holds them.
    let mut known = [CardSet::EMPTY; 3];
    for (c, seat) in view.known_holdings() {
        known[rel(view, seat) - 1].insert(c);
    }
    for k in known {
        e.set(k);
    }

    let mine = rank_counts(hand);
    for r in 2..=14 {
        e.frac(mine[r] as f32, 4.0);
    }
    let unseen = view.unseen();
    let u = rank_counts(unseen);
    for r in 2..=14 {
        e.frac(u[r] as f32, 4.0);
    }
    for c in [MAH_JONG, DOG, PHOENIX, DRAGON] {
        e.bit(unseen.contains(c));
    }

    // Players, in relative order.
    let mut s = view.seat;
    for _ in 0..4 {
        let p = &view.players[s];
        e.frac(p.card_count as f32, 14.0);
        e.bit(p.is_out());
        e.one_hot((p.out_order as usize).checked_sub(1), 3);
        e.bit(p.tichu == TichuCall::Small);
        e.bit(p.tichu == TichuCall::Grand);
        e.bit(p.has_played_first_card);
        e.frac(p.trick_count as f32, 14.0);
        e.range(p.captured_points as f32, -25.0, 125.0);
        e.bit(view.last_played_by == Some(s));
        e.bit(view.turn == s);
        s = view.next_seat(s);
    }

    // The trick.
    e.bit(view.is_leading());
    e.one_hot(view.current_trick.as_ref().map(|t| kind_index(t.kind)), 8);
    let t = view.current_trick.as_ref();
    e.frac(t.map_or(0.0, |t| t.rank), 16.0);
    e.frac(t.map_or(0.0, |t| t.length as f32), 14.0);
    e.range(view.trick_points() as f32, -25.0, 100.0);
    e.frac(view.pass_count as f32, 3.0);
    e.frac(view.trick_plays.len() as f32, 10.0);

    // The wish.
    e.bit(view.wish.is_some());
    e.one_hot(view.wish.map(|r| r as usize - 2), 13);

    // Game score (0-0 in round mode).
    let team = view.my_team();
    e.range(view.scores[team] as f32, -500.0, 1500.0);
    e.range(view.scores[1 - team] as f32, -500.0, 1500.0);
    e.frac(bomb_count(hand) as f32, 3.0);

    // --- Version 2 ---

    // The card each seat passed me (by rank class).
    for r in 1..4 {
        let got = view.received.iter().find(|(_, from)| rel(view, *from) == r).map(|(c, _)| rank_class(*c));
        e.one_hot(got, 17);
    }

    // What the others passed on, and what everyone led.
    let h = history(view);
    for r in 1..4 {
        for &rank in &h.passed_on[r] {
            e.bit(rank > 0.0);
            e.frac(rank, 16.0);
        }
    }
    for r in 0..4 {
        for &n in &h.leads[r] {
            e.frac(n as f32, 4.0);
        }
    }
    e.frac(h.tricks as f32, 20.0);

    // Bombs the others might hold, and mine.
    let threats = bomb_threats(view);
    for t in &threats[1..] {
        e.frac(t.quads, 2.0);
        e.frac(t.any, 1.0);
        e.frac(t.best, 1.0);
        e.bit(t.best > 0.0);
    }
    e.frac(best_bomb(hand), 1.0);

    // How my hand plays out (h2's plan, with the unseen cards as opponents).
    let mine = Counts::of(view.hand.iter().copied());
    let uc = Counts::of(unseen.cards());
    let cost = h2::cost(&mine);
    let plan = h2::plan(&mine, &uc, cost);
    e.frac(cost as f32, 14.0);
    e.frac(plan.lead_bosses as f32, 8.0);
    e.frac(plan.regain as f32, 8.0);
    e.frac(plan.losers as f32, 14.0);
    e.bit(h2::clear_path(&mine, &uc));

    // Can the trick on the table still be beaten (bombs aside)?
    e.bit(t.is_some_and(|t| h2::could_beat(&h2::rcombo(t), &uc)));
    e.bit(view.last_played_by == Some(view.partner()) && t.is_some());

    let n = out.len() - start;
    assert!(n <= STATE_DIM, "state features overflow: {n}");
    out.resize(start + STATE_DIM, 0);
}

/// What every candidate's features share at one decision.
pub struct ActionContext {
    unseen: Counts,
    /// The strongest bomb an opponent could hold.
    opp_best_bomb: f32,
}

impl ActionContext {
    pub fn new(view: &PlayerView) -> ActionContext {
        let threats = bomb_threats(view);
        ActionContext {
            unseen: Counts::of(view.unseen().cards()),
            opp_best_bomb: threats[1].best.max(threats[3].best),
        }
    }
}

/// Encode one candidate: `Some(play)`, or `None` for passing.
pub fn encode_action(view: &PlayerView, play: Option<&Combo>, out: &mut Vec<u8>) {
    encode_action_in(&ActionContext::new(view), view, play, out)
}

/// Unseen ranks above `c` that could make the same single, pair or triple.
fn beaters(c: &Combo, u: &Counts) -> usize {
    let need = match c.kind {
        ComboKind::Single => 1,
        ComboKind::Pair => 2,
        ComboKind::Triple => 3,
        _ => return 0,
    };
    let above = (c.rank.floor() as usize + 1).max(2);
    let ranks = (above..=14).filter(|&q| u.n[q] + u.phoenix as u8 >= need).count();
    ranks + (need == 1 && u.dragon && c.rank < 15.0) as usize
}

/// `encode_action` with the shared part computed once per decision.
pub fn encode_action_in(ctx: &ActionContext, view: &PlayerView, play: Option<&Combo>, out: &mut Vec<u8>) {
    let start = out.len();
    let mut e = Enc(out);
    let hand = CardSet::from_cards(&view.hand);
    e.bit(play.is_none());
    let used = play.map_or(CardSet::EMPTY, |c| c.card_set());
    let after = hand.minus(used);
    e.one_hot(play.map(|c| kind_index(c.kind)), 8);
    e.frac(play.map_or(0.0, |c| c.rank), 16.0);
    e.frac(play.map_or(0.0, |c| c.length as f32), 14.0);
    let uc = rank_counts(used);
    for r in 2..=14 {
        e.frac(uc[r] as f32, 4.0);
    }
    for c in [MAH_JONG, DOG, PHOENIX, DRAGON] {
        e.bit(used.contains(c));
    }
    e.range(play.map_or(0, |c| total_points(&c.cards)) as f32, -25.0, 100.0);
    e.frac(after.len() as f32, 14.0);
    e.bit(play.is_some() && after.is_empty());
    let hc = rank_counts(hand);
    let split = (2..=14).filter(|&r| uc[r] > 0 && uc[r] < hc[r]).count();
    e.frac(split as f32, 4.0);
    e.frac(bomb_count(hand).saturating_sub(bomb_count(after)) as f32, 2.0);
    let ac = rank_counts(after);
    e.frac((2..=14).filter(|&r| ac[r] == 1).count() as f32, 13.0);
    e.frac((2..=10).filter(|&r| ac[r] == 1).count() as f32, 9.0);
    e.bit(play.is_some_and(|c| view.wish.is_some_and(|w| c.contains_rank(w))));
    e.bit(play.is_some() && view.last_played_by == Some(view.partner()));
    e.bit(play.is_some_and(|c| c.is_bomb()));

    // --- Version 2 ---

    // Can it be beaten? Bombs: by a stronger bomb an opponent might hold.
    let boss = play.is_some_and(|c| {
        if c.is_bomb() {
            bomb_strength(c.kind, c.rank, c.length) > ctx.opp_best_bomb
        } else {
            !h2::could_beat(&h2::rcombo(c), &ctx.unseen)
        }
    });
    e.bit(boss);
    e.frac(play.map_or(0, |c| beaters(c, &ctx.unseen)) as f32, 14.0);

    // How the rest of my hand plays out after it.
    let rest = Counts::of(after.cards());
    let cost = h2::cost(&rest);
    let plan = h2::plan(&rest, &ctx.unseen, cost);
    e.frac(cost as f32, 14.0);
    e.frac(plan.lead_bosses as f32, 8.0);
    e.frac(plan.regain as f32, 8.0);
    e.frac(plan.losers as f32, 14.0);
    e.bit(h2::clear_path(&rest, &ctx.unseen));

    let n = out.len() - start;
    assert!(n <= ACTION_DIM, "action features overflow: {n}");
    out.resize(start + ACTION_DIM, 0);
}

/// The options at a play decision, collapsed to one per distinct feature
/// vector (plays that differ only in suits look the same to a net).
pub struct Candidates {
    /// `None` = pass, `Some(i)` = `plays[i]` (the first of its group).
    pub options: Vec<Option<usize>>,
    /// `ACTION_DIM` bytes per option.
    pub features: Vec<u8>,
    /// For each play index, the option it collapsed into.
    pub group_of: Vec<usize>,
}

impl Candidates {
    pub fn new(view: &PlayerView, plays: &[Combo], can_pass: bool) -> Candidates {
        let mut c = Candidates { options: vec![], features: vec![], group_of: vec![] };
        let mut seen: std::collections::HashMap<Vec<u8>, usize> = Default::default();
        let mut buf = Vec::with_capacity(ACTION_DIM);
        let ctx = ActionContext::new(view);
        let opts = plays.iter().enumerate().map(|(i, p)| (Some(i), Some(p)));
        let pass = can_pass.then_some((None, None));
        for (idx, play) in opts.chain(pass) {
            buf.clear();
            encode_action_in(&ctx, view, play, &mut buf);
            let g = *seen.entry(buf.clone()).or_insert_with(|| {
                c.options.push(idx);
                c.features.extend_from_slice(&buf);
                c.options.len() - 1
            });
            if idx.is_some() {
                c.group_of.push(g);
            }
        }
        c
    }

    pub fn len(&self) -> usize {
        self.options.len()
    }

    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    /// The option a bot's answer (`Bot::play`'s return value) belongs to.
    pub fn option_of(&self, choice: Option<usize>) -> Option<usize> {
        match choice {
            Some(i) => self.group_of.get(i).copied(),
            None => self.options.iter().position(|o| o.is_none()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::HeuristicBot;
    use crate::bot::Bot;
    use crate::deck::shuffled_deck;
    use crate::engine::GameState;
    use crate::sim::play_dealt_round;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Encodes every play decision of some heuristic rounds without overflow,
    /// and every play maps to an option.
    #[test]
    fn encodes_real_positions() {
        struct Check(HeuristicBot, usize);
        impl Bot for Check {
            fn name(&self) -> String {
                "check".into()
            }
            fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
                self.0.pass_cards(v)
            }
            fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
                self.0.dragon_recipient(v)
            }
            fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
                let mut s = vec![];
                encode_state(v, &mut s);
                assert_eq!(s.len(), STATE_DIM);
                let c = Candidates::new(v, plays, can_pass);
                assert_eq!(c.features.len(), c.len() * ACTION_DIM);
                assert_eq!(c.group_of.len(), plays.len());
                let choice = self.0.play(v, plays, can_pass);
                let o = c.option_of(choice).unwrap();
                assert_eq!(c.option_of(c.options[o]), Some(o));
                self.1 += 1;
                choice
            }
        }
        let mut rng = StdRng::seed_from_u64(7);
        for i in 0..50 {
            let settings = Settings { clockwise: i % 2 == 1, ..Settings::default() };
            let mut state = GameState::new(settings);
            state.start_round(shuffled_deck(&mut rng));
            let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|_| Box::new(Check(HeuristicBot::new(), 0)) as Box<dyn Bot>);
            play_dealt_round(&mut state, &mut bots, None).unwrap();
        }
    }

    /// Seat 0's view at the start of play, to be edited by the tests below.
    fn playing_view() -> PlayerView {
        let mut state = GameState::new(Settings { clockwise: true, ..Settings::default() });
        state.start_round(shuffled_deck(&mut StdRng::seed_from_u64(5)));
        for s in 0..4 {
            state.call_grand_tichu(s, false).unwrap();
        }
        for s in 0..4 {
            let pass = HeuristicBot::new().pass_cards(&state.view(s));
            state.pass_cards(s, pass).unwrap();
        }
        state.view(0)
    }

    fn c(suit: Suit, rank: Rank) -> Card {
        Card::normal(suit, rank)
    }

    /// Sixes and jacks of every suit block every straight flush.
    fn sixes_and_jacks() -> Vec<Card> {
        Suit::ALL.iter().flat_map(|&s| [c(s, 6), c(s, 11)]).collect()
    }

    #[test]
    fn no_bomb_threat_when_every_bomb_is_ruled_out() {
        let mut v = playing_view();
        v.my_pass = None;
        v.hand = sixes_and_jacks();
        // One card of every other rank is gone.
        v.played_cards = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14].iter().map(|&r| c(Suit::Jade, r)).collect();
        for (s, n) in [(1, 13), (2, 12), (3, 12)] {
            v.players[s].card_count = n;
        }
        for t in &bomb_threats(&v)[1..] {
            assert_eq!((t.quads, t.any, t.best), (0.0, 0.0, 0.0));
        }
    }

    #[test]
    fn a_passed_card_pins_the_bomb_to_its_recipient() {
        let mut v = playing_view();
        v.hand = sixes_and_jacks();
        // Every rank but the nines has a card gone; I passed the nine of
        // swords (to seat 3, my left), and two cards that have been played.
        v.played_cards = [2, 3, 4, 5, 7, 8, 10, 12, 13, 14].iter().map(|&r| c(Suit::Jade, r)).collect();
        v.my_pass = Some(PassInfo { left: c(Suit::Sword, 9), partner: c(Suit::Jade, 2), right: c(Suit::Jade, 3) });
        for (s, n) in [(1, 13), (2, 12), (3, 13)] {
            v.players[s].card_count = n;
        }
        let t = bomb_threats(&v);
        let holder = rel(&v, 3);
        for r in 1..4 {
            if r == holder {
                // The other three nines among its 12 unknown cards, out of 37.
                let want = (12.0 * 11.0 * 10.0) / (37.0 * 36.0 * 35.0);
                assert!((t[r].quads - want).abs() < 1e-6, "{} vs {want}", t[r].quads);
                assert!((t[r].any - want).abs() < 1e-6);
                assert!(t[r].best > 0.0);
            } else {
                assert_eq!((t[r].quads, t[r].best), (0.0, 0.0));
            }
        }
    }

    #[test]
    fn history_tracks_leads_and_passes_against_opponents() {
        let mut v = playing_view();
        let (a, b, p) = (1, 2, 3); // a and p are partners; b is their opponent
        v.log = vec![
            Action::Play { seat: a, cards: vec![c(Suit::Jade, 5), c(Suit::Sword, 5)], phoenix_as: None },
            Action::PassTurn { seat: b },
            Action::PassTurn { seat: p },
            Action::ResolveTrick,
            Action::Play { seat: b, cards: vec![DOG], phoenix_as: None },
        ];
        let h = history(&v);
        assert_eq!(h.tricks, 1);
        assert_eq!(h.leads[rel(&v, a)][kind_index(ComboKind::Pair)], 1);
        assert_eq!(h.leads[rel(&v, b)][6], 1);
        assert_eq!(h.passed_on[rel(&v, b)][kind_index(ComboKind::Pair)], 5.0);
        assert_eq!(h.passed_on[rel(&v, p)], [0.0; 6]);
    }

    /// Predicted P(seat holds a bomb) against what the seat really holds, over
    /// heuristic deals, in deciles (`MAX_PLAYED=n` keeps only decisions with
    /// at most n cards played). Also prints how often a hand holds a bomb
    /// before and after the pass: passing roughly doubles it, which the
    /// uniform model can't know, so expect it to under-predict while still
    /// ranking seats correctly. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn bomb_threats_are_calibrated() {
        use std::cell::RefCell;
        use std::rc::Rc;
        type Hands = Rc<RefCell<[Vec<Card>; 4]>>;
        type Stats = Rc<RefCell<Vec<(f32, bool)>>>;
        struct Probe(HeuristicBot, Hands, Stats, usize);
        impl Bot for Probe {
            fn name(&self) -> String {
                "probe".into()
            }
            fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
                self.0.pass_cards(v)
            }
            fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
                self.0.dragon_recipient(v)
            }
            fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
                if v.played_cards.len() <= self.3 {
                    let t = bomb_threats(v);
                    for x in (0..4).filter(|&x| x != v.seat && !v.players[x].is_out()) {
                        let mut held = CardSet::from_cards(&self.1.borrow()[x]);
                        for a in &v.log {
                            if let Action::Play { seat, cards, .. } | Action::Bomb { seat, cards } = a {
                                if *seat == x {
                                    cards.iter().for_each(|c| held.remove(*c));
                                }
                            }
                        }
                        self.2.borrow_mut().push((t[rel(v, x)].any, bomb_count(held) > 0));
                    }
                }
                self.0.play(v, plays, can_pass)
            }
        }
        let max_played = std::env::var("MAX_PLAYED").ok().and_then(|s| s.parse().ok()).unwrap_or(99);
        let with_bomb = |s: &GameState| (0..4).filter(|&p| bomb_count(CardSet::from_cards(&s.players[p].hand)) > 0).count();
        let (mut before, mut after) = (0, 0);
        let stats: Stats = Rc::new(RefCell::new(vec![]));
        let mut rng = StdRng::seed_from_u64(11);
        let deals = 2000;
        for i in 0..deals {
            let mut state = GameState::new(Settings { clockwise: i % 2 == 1, ..Settings::default() });
            state.start_round(shuffled_deck(&mut rng));
            for s in 0..4 {
                state.call_grand_tichu(s, false).unwrap();
            }
            before += with_bomb(&state);
            for s in 0..4 {
                let pass = HeuristicBot::new().pass_cards(&state.view(s));
                state.pass_cards(s, pass).unwrap();
            }
            after += with_bomb(&state);
            let hands: Hands = Rc::new(RefCell::new([0, 1, 2, 3].map(|s| state.players[s].hand.clone())));
            let mut bots: [Box<dyn Bot>; 4] = [0, 1, 2, 3]
                .map(|_| Box::new(Probe(HeuristicBot::new(), hands.clone(), stats.clone(), max_played)) as Box<dyn Bot>);
            crate::sim::play_from(&mut state, &mut bots, None, false).unwrap();
        }
        let hands = (4 * deals) as f64;
        println!("hands with a bomb: {:.3} before the pass, {:.3} after", before as f64 / hands, after as f64 / hands);
        let stats = stats.borrow();
        let n = stats.len() as f64;
        let pred = stats.iter().map(|s| s.0 as f64).sum::<f64>() / n;
        let real = stats.iter().filter(|s| s.1).count() as f64 / n;
        println!("{n} samples: mean predicted {pred:.4}, actual {real:.4}");
        for d in 0..10 {
            let (lo, hi) = (d as f32 / 10.0, (d + 1) as f32 / 10.0);
            let b: Vec<_> = stats.iter().filter(|s| s.0 >= lo && (s.0 < hi || d == 9)).collect();
            if !b.is_empty() {
                let p = b.iter().map(|s| s.0 as f64).sum::<f64>() / b.len() as f64;
                let a = b.iter().filter(|s| s.1).count() as f64 / b.len() as f64;
                println!("  [{lo:.1}, {hi:.1}): {:>7} samples, predicted {p:.3}, actual {a:.3}", b.len());
            }
        }
    }

    #[test]
    fn relative_seats_follow_play_order() {
        let mut state = GameState::new(Settings { clockwise: true, ..Settings::default() });
        state.start_round(shuffled_deck(&mut StdRng::seed_from_u64(1)));
        let v = state.view(0);
        assert_eq!(rel(&v, 0), 0);
        assert_eq!(rel(&v, 3), 1);
        assert_eq!(rel(&v, 2), 2);
        assert_eq!(rel(&v, 1), 3);
    }
}
