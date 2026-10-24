//! Fixed-size features of a `PlayerView` and of each candidate play, for
//! learned bots (`bots::net`) and the self-play exporter (`selfplay`).
//!
//! Policy nets score each candidate separately (`state ⊕ action → logit`),
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

use crate::engine::Action;
use crate::scoring::total_points;
use crate::types::*;
use crate::view::PlayerView;

pub const FEATURE_VERSION: u32 = 1;
pub const STATE_DIM: usize = 560;
pub const ACTION_DIM: usize = 40;

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

    let n = out.len() - start;
    assert!(n <= STATE_DIM, "state features overflow: {n}");
    out.resize(start + STATE_DIM, 0);
}

/// Encode one candidate: `Some(play)`, or `None` for passing.
pub fn encode_action(view: &PlayerView, play: Option<&Combo>, out: &mut Vec<u8>) {
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
        let opts = plays.iter().enumerate().map(|(i, p)| (Some(i), Some(p)));
        let pass = can_pass.then_some((None, None));
        for (idx, play) in opts.chain(pass) {
            buf.clear();
            encode_action(view, play, &mut buf);
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
