//! Exact solver for the two-player endgame.
//!
//! Once only two players hold cards (necessarily opponents: two partners out
//! first ends the round as a 1-2), each can deduce the other's hand — it is
//! every card not in their own hand and not yet played. So the rest of the
//! round is a perfect-information game, and alpha-beta search solves it.
//!
//! The search follows the simulator's decision points exactly (see `sim`):
//! after every play or bomb the other seats are offered a bomb in play order
//! (skipping the player who just played), and during the trick countdown
//! bombs are offered starting after the winner and ending with them. Those
//! offers aren't engine states, so they're explicit `Node`s here. All
//! transitions go through the engine, so the rules can't drift.

use std::collections::HashMap;

use crate::engine::{Action, GameState, Pending};
use crate::types::*;

/// Where in the decision sequence a state is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Node {
    /// Whatever `GameState::pending()` says.
    Engine,
    /// A bomb offer to `seats[0]`, then (if declined) the rest in order.
    /// `countdown`: the trick-countdown offer, which ends by awarding the
    /// trick; otherwise the offer after a play, which ends by moving on.
    Offer { countdown: bool, seats: [u8; 4], len: u8 },
}

impl Node {
    fn offer(countdown: bool, seats: &[Seat]) -> Node {
        let mut s = [0u8; 4];
        for (i, &x) in seats.iter().enumerate() {
            s[i] = x as u8;
        }
        Node::Offer { countdown, seats: s, len: seats.len() as u8 }
    }

    /// The offer the simulator makes after `by` plays or bombs.
    pub fn after_play(state: &GameState, by: Seat) -> Node {
        let mut seats = vec![];
        let mut s = state.next_seat(by);
        for _ in 0..3 {
            seats.push(s);
            s = state.next_seat(s);
        }
        Node::offer(false, &seats)
    }

    /// The offer the simulator makes when the countdown for `winner` starts.
    pub fn countdown(state: &GameState, winner: Seat) -> Node {
        let mut seats = vec![];
        let mut s = state.next_seat(winner);
        for _ in 0..4 {
            seats.push(s);
            s = state.next_seat(s);
        }
        Node::offer(true, &seats)
    }

    /// The same offer, but starting at `seat` (the seats before it declined).
    pub fn from_seat(self, seat: Seat) -> Node {
        match self {
            Node::Offer { countdown, seats, len } => {
                let list: Vec<Seat> = seats[..len as usize].iter().map(|&s| s as Seat).collect();
                let i = list.iter().position(|&s| s == seat).expect("seat not in offer");
                Node::offer(countdown, &list[i..])
            }
            Node::Engine => Node::Engine,
        }
    }
}

/// A choice at a decision node.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Play(Combo),
    Pass,
    Bomb(Combo),
    NoBomb,
    Wish(Option<Rank>),
    GiveDragon(Seat),
}

#[derive(Debug, Clone)]
pub struct Solution {
    /// Round margin (team 0's round total minus team 1's) under best play.
    pub value: i32,
    /// The seat to decide at the root, and its best choice.
    pub seat: Seat,
    pub decision: Decision,
    pub nodes: usize,
}

impl Solution {
    /// The value from `team`'s side.
    pub fn value_for(&self, team: Team) -> i32 {
        if team == 0 { self.value } else { -self.value }
    }
}

/// Exactly two players still hold cards, and the round is being played.
pub fn is_two_player_endgame(state: &GameState) -> bool {
    state.phase == Phase::Playing && state.players.iter().filter(|p| !p.is_out()).count() == 2
}

/// Solve from `state` at its engine-pending decision (a trick countdown
/// starts with the countdown bomb offer). None if it isn't a two-player
/// endgame, there's no decision to make, or the search exceeds `budget` nodes.
pub fn solve(state: &GameState, budget: usize) -> Option<Solution> {
    solve_node(state, Node::Engine, budget)
}

/// Solve from `state` at `node`.
pub fn solve_node(state: &GameState, node: Node, budget: usize) -> Option<Solution> {
    run(state, node, budget, false)
}

/// The same search without pruning, transposition table, or merging of
/// suit-only variants: slow, for checking `solve_node` in tests.
pub fn solve_reference(state: &GameState, node: Node, budget: usize) -> Option<Solution> {
    run(state, node, budget, true)
}

fn run(state: &GameState, node: Node, budget: usize, reference: bool) -> Option<Solution> {
    if !is_two_player_endgame(state) {
        return None;
    }
    let mut root = state.clone();
    root.log.clear();
    root.round_results.clear();
    root.deck.clear();
    let (root, node) = normalize(root, node);
    let node = node?;
    let mut s = Searcher { tt: HashMap::new(), nodes: 0, budget, reference, root_best: 0 };
    let value = s.search(&root, node, i32::MIN + 1, i32::MAX - 1, true).ok()?;
    let (seat, moves) = moves(&root, node, !reference);
    let decision = moves.into_iter().nth(s.root_best)?.decision;
    Some(Solution { value, seat, decision, nodes: s.nodes })
}

/// The exact value of every option at a decision.
#[derive(Debug, Clone)]
pub struct OptionValue {
    pub decision: Decision,
    /// The engine action (None for declining a bomb offer).
    pub action: Option<Action>,
    /// Round margin for team 0 under best play after this choice.
    pub value: i32,
}

/// Solve every option at `node` (no suit merging at the root, so each legal
/// play is listed). Returns the deciding seat and the options, or None if it
/// isn't a two-player endgame decision or the search runs over `budget`
/// nodes in total.
pub fn solve_options(state: &GameState, node: Node, budget: usize) -> Option<(Seat, Vec<OptionValue>)> {
    if !is_two_player_endgame(state) {
        return None;
    }
    let mut root = state.clone();
    root.log.clear();
    root.round_results.clear();
    root.deck.clear();
    let (root, node) = normalize(root, node);
    let node = node?;
    let (seat, moves) = moves(&root, node, false);
    let mut s = Searcher { tt: HashMap::new(), nodes: 0, budget, reference: false, root_best: 0 };
    let mut out = vec![];
    for m in moves {
        let mut child = root.clone();
        if let Some(a) = &m.action {
            child.apply(a).ok()?;
        }
        let (child, next) = normalize(child, m.next);
        let value = match next {
            None => child.last_result()?.diff(),
            Some(n) => s.search(&child, n, i32::MIN + 1, i32::MAX - 1, false).ok()?,
        };
        out.push(OptionValue { decision: m.decision, action: m.action, value });
    }
    Some((seat, out))
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

struct Move {
    decision: Decision,
    action: Option<Action>,
    next: Node,
    /// Ordering score: higher is searched first.
    order: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

struct Entry {
    value: i32,
    bound: Bound,
    best: u16,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    hands: [u64; 4],
    trick: Option<(ComboKind, i32, u8, bool)>,
    trick_points: i32,
    captured: [i32; 4],
    last: u8,
    turn: u8,
    pass_count: u8,
    wish: u8,
    wish_pending: bool,
    countdown: u8,
    dragon_by: u8,
    out: [u8; 4],
    node: Node,
}

struct Abort;

struct Searcher {
    tt: HashMap<Key, Entry>,
    nodes: usize,
    budget: usize,
    reference: bool,
    root_best: usize,
}

fn seat_code(s: Option<Seat>) -> u8 {
    s.map(|x| x as u8).unwrap_or(9)
}

/// Can suits in this hand ever matter? Only for straight-flush bombs, which
/// need five cards of one suit.
fn suits_matter(hand: &[Card]) -> bool {
    let mut n = [0u8; 4];
    for c in hand {
        if let Some(s) = c.suit() {
            n[s.index()] += 1;
        }
    }
    n.iter().any(|&k| k >= 5)
}

/// A hand's identity for the table. When suits can't matter, hands that
/// differ only in suits are the same position.
fn hand_code(hand: &[Card]) -> u64 {
    if suits_matter(hand) {
        return CardSet::from_cards(hand).0;
    }
    let mut code = 1u64 << 63;
    for c in hand {
        match c.rank() {
            Some(r) => code += 1u64 << (4 * (r as u64 - 2)),
            None => code |= 1u64 << (52 + (c.index() as u64 - 52)),
        }
    }
    code
}

fn key(st: &GameState, node: Node) -> Key {
    let trick_points: i32 = st.trick_plays.iter().map(|p| crate::scoring::total_points(&p.combo.cards)).sum();
    Key {
        hands: [0, 1, 2, 3].map(|s| hand_code(&st.players[s].hand)),
        trick: st
            .current_trick
            .as_ref()
            .map(|t| (t.kind, (t.rank * 2.0) as i32, t.length as u8, t.contains(SpecialKind::Dragon))),
        trick_points,
        captured: [0, 1, 2, 3].map(|s| st.players[s].captured_points()),
        last: seat_code(st.last_played_by),
        turn: st.turn as u8,
        pass_count: st.pass_count as u8,
        wish: st.wish.unwrap_or(0),
        wish_pending: st.wish_pending,
        countdown: seat_code(st.trick_countdown),
        dragon_by: seat_code(st.dragon_giveaway_by),
        out: [0, 1, 2, 3].map(|s| st.players[s].out_order),
        node,
    }
}

/// Advance through steps nobody decides (offers to seats without bombs,
/// awarding an un-bombed trick) to the next decision. None = round over.
fn normalize(mut st: GameState, mut node: Node) -> (GameState, Option<Node>) {
    loop {
        match node {
            Node::Engine => match st.pending() {
                Pending::TrickCountdown(w) => node = Node::countdown(&st, w),
                Pending::RoundOver | Pending::GameOver => return (st, None),
                _ => return (st, Some(node)),
            },
            Node::Offer { countdown, seats, len } => {
                if st.phase != Phase::Playing {
                    return (st, None);
                }
                if !countdown && (st.wish_pending || st.current_trick.is_none()) {
                    node = Node::Engine;
                    continue;
                }
                let list = &seats[..len as usize];
                match list.iter().position(|&s| !st.legal_bombs(s as Seat).is_empty()) {
                    Some(i) => {
                        let rest: Vec<Seat> = list[i..].iter().map(|&s| s as Seat).collect();
                        return (st, Some(Node::offer(countdown, &rest)));
                    }
                    None => {
                        if countdown {
                            st.resolve_trick().expect("resolve during countdown");
                        }
                        node = Node::Engine;
                    }
                }
            }
        }
    }
}

/// The seat to decide at `node` (a normalized decision node) and its options.
/// With `collapse`, plays that differ only in suits (when suits can't matter
/// for this hand) are listed once.
fn moves(st: &GameState, node: Node, collapse: bool) -> (Seat, Vec<Move>) {
    match node {
        Node::Offer { countdown, seats, len } => {
            let seat = seats[0] as Seat;
            let rest: Vec<Seat> = seats[1..len as usize].iter().map(|&s| s as Seat).collect();
            let mut out: Vec<Move> = st
                .legal_bombs(seat)
                .into_iter()
                .map(|b| Move {
                    action: Some(Action::Bomb { seat, cards: b.cards.clone() }),
                    next: Node::after_play(st, seat),
                    order: b.length as i32 * 10 - b.rank as i32,
                    decision: Decision::Bomb(b),
                })
                .collect();
            out.push(Move { decision: Decision::NoBomb, action: None, next: Node::offer(countdown, &rest), order: 100 });
            (seat, out)
        }
        Node::Engine => match st.pending() {
            Pending::Turn(seat) => {
                let hand = &st.players[seat].hand;
                let collapse = collapse && !suits_matter(hand);
                let mut seen = std::collections::HashSet::new();
                let mut out = vec![];
                for c in st.legal_plays(seat) {
                    if collapse && !c.is_bomb() {
                        let sig = hand_code(&c.cards);
                        if !seen.insert((sig, (c.rank * 2.0) as i32)) {
                            continue;
                        }
                    }
                    let goes_out = c.cards.len() == hand.len();
                    let order = if goes_out { 1000 } else { 0 } + c.cards.len() as i32 * 10
                        - if c.is_bomb() { 50 } else { 0 }
                        - c.rank as i32;
                    out.push(Move {
                        action: Some(Action::Play { seat, cards: c.cards.clone(), phoenix_as: c.phoenix_as }),
                        next: Node::after_play(st, seat),
                        order,
                        decision: Decision::Play(c),
                    });
                }
                if st.can_pass(seat) {
                    out.push(Move {
                        decision: Decision::Pass,
                        action: Some(Action::PassTurn { seat }),
                        next: Node::Engine,
                        order: -500,
                    });
                }
                (seat, out)
            }
            Pending::Wish(seat) => {
                // A rank nobody holds binds nobody: the same as no wish.
                let mut ranks: Vec<Rank> = st
                    .players
                    .iter()
                    .flat_map(|p| p.hand.iter().filter_map(|c| c.rank()))
                    .collect();
                ranks.sort_unstable();
                ranks.dedup();
                let mut out = vec![Move {
                    decision: Decision::Wish(None),
                    action: Some(Action::Wish { seat, rank: None }),
                    next: Node::Engine,
                    order: 0,
                }];
                for r in ranks {
                    out.push(Move {
                        decision: Decision::Wish(Some(r)),
                        action: Some(Action::Wish { seat, rank: Some(r) }),
                        next: Node::Engine,
                        order: 0,
                    });
                }
                (seat, out)
            }
            Pending::DragonGiveaway(seat) => {
                let out = opponents(seat)
                    .into_iter()
                    .map(|to| Move {
                        decision: Decision::GiveDragon(to),
                        action: Some(Action::GiveDragon { seat, to }),
                        next: Node::Engine,
                        order: 0,
                    })
                    .collect();
                (seat, out)
            }
            p => panic!("not a decision node: {p:?}"),
        },
    }
}

impl Searcher {
    /// Minimax value (team 0's margin) of a normalized decision node.
    fn search(&mut self, st: &GameState, node: Node, mut alpha: i32, mut beta: i32, root: bool) -> Result<i32, Abort> {
        self.nodes += 1;
        if self.nodes > self.budget {
            return Err(Abort);
        }
        let k = key(st, node);
        let mut tt_best = None;
        if let Some(e) = self.tt.get(&k).filter(|_| !root) {
            match e.bound {
                Bound::Exact => return Ok(e.value),
                Bound::Lower => alpha = alpha.max(e.value),
                Bound::Upper => beta = beta.min(e.value),
            }
            if alpha >= beta {
                return Ok(e.value);
            }
            tt_best = Some(e.best as usize);
        }
        let (seat, moves) = moves(st, node, !self.reference);
        let maximize = team_for_seat(seat) == 0;
        let mut order: Vec<usize> = (0..moves.len()).collect();
        order.sort_by_key(|&i| -moves[i].order);
        if let Some(b) = tt_best {
            if let Some(pos) = order.iter().position(|&i| i == b) {
                order.remove(pos);
                order.insert(0, b);
            }
        }
        let (a0, b0) = (alpha, beta);
        let mut best_val = if maximize { i32::MIN } else { i32::MAX };
        let mut best_i = order[0];
        for i in order {
            let m = &moves[i];
            let mut child = st.clone();
            if let Some(a) = &m.action {
                child.apply(a).expect("solver move rejected by engine");
            }
            let (child, next) = normalize(child, m.next);
            let v = match next {
                None => child.last_result().expect("round over without result").diff(),
                Some(n) => self.search(&child, n, alpha, beta, false)?,
            };
            if (maximize && v > best_val) || (!maximize && v < best_val) {
                best_val = v;
                best_i = i;
            }
            if maximize {
                alpha = alpha.max(v);
            } else {
                beta = beta.min(v);
            }
            if alpha >= beta && !self.reference {
                break;
            }
        }
        if root {
            self.root_best = best_i;
        }
        if self.reference {
            return Ok(best_val);
        }
        let bound = if best_val <= a0 {
            Bound::Upper
        } else if best_val >= b0 {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt.insert(k, Entry { value: best_val, bound, best: best_i as u16 });
        Ok(best_val)
    }
}
