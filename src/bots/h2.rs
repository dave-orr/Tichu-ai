//! `h2`: a stronger rule-based player, meant both to play and to drive pimc
//! rollouts (so it has to stay fast).
//!
//! Over `heuristic` it adds:
//! - **Card tracking.** A combination is a *boss* if nothing in the unseen
//!   cards could beat it (bombs aside). Bosses are the ways back to the lead.
//! - **Hand structure.** A fast rank-level split of the hand into the fewest
//!   combinations (`Shape`). Plays are judged by how many combinations the
//!   rest of the hand needs, so the bot doesn't break pairs and straights to
//!   follow cheaply.
//! - **Lead planning.** With at most one loser left, cash bosses and go out
//!   on the loser; otherwise lead the cheapest loser, avoiding the lengths a
//!   near-out opponent could follow.
//! - **Defense and support.** Fight (bosses, broken combos, bombs) when the
//!   trick goes to an opponent who called Tichu, is nearly out, or is chasing
//!   a 1-2; push for our own 1-2 when the partner went out first; feed a
//!   partner who is nearly out.
//! - **Dave's rules.** Lead the Mah Jong first; wish for an Ace when the next
//!   player called, else for the card I passed the next player; play over a
//!   partner's low single or pair unless they called (as the caller, play
//!   over them with any loser); give the Dragon to an opponent who didn't
//!   call.

use crate::bot::Bot;
use crate::bots::calls;
use crate::bots::HeuristicBot;
use crate::types::*;
use crate::view::PlayerView;

pub struct H2Bot {
    base: HeuristicBot,
    pub p: Params,
}

/// Tunable knobs (`h2~name=value,...` on the command line).
#[derive(Debug, Clone)]
pub struct Params {
    /// Trick points worth fighting for with anything.
    pub fight_pts: f32,
    /// Trick points worth spending a boss on.
    pub boss_pts: f32,
    /// Opponent card count at or below which leads avoid short combos.
    pub danger: f32,
    /// Fight for every trick while my own Tichu is live (0/1).
    pub tichu_fight: f32,
    /// Out-of-turn bombs: 1 = h2 rules, 0 = the heuristic's.
    pub bomb_mode: f32,
    /// Lead bonus for singles when partner's Tichu is live.
    pub partner_single: f32,
    /// Lead bonus per card shed.
    pub shed: f32,
    /// Lead the Mah Jong (alone or in a run) as soon as I have the lead (0/1).
    pub mj_first: f32,
    /// Wish: 1 = an Ace if the next player called, else the card I passed
    /// the next player; 0 = the heuristic's (the highest card I passed an opponent).
    pub wish_mode: f32,
    /// Play over a partner's low single (below 10) or low pair (below 8s)
    /// unless they called; as the caller, play over them with any loser (0/1).
    pub overtake: f32,
    /// Dragon gift: 1 = an opponent without a call first, then more cards;
    /// 0 = more cards.
    pub dragon_mode: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            fight_pts: 15.0,
            boss_pts: 10.0,
            danger: 0.0,
            tichu_fight: 1.0,
            bomb_mode: 1.0,
            partner_single: 3.0,
            shed: 1.5,
            mj_first: 1.0,
            wish_mode: 1.0,
            overtake: 1.0,
            dragon_mode: 1.0,
        }
    }
}

impl Params {
    pub fn parse(spec: &str) -> Option<Params> {
        let mut p = Params::default();
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=')?;
            let v: f32 = v.parse().ok()?;
            match k {
                "fight_pts" => p.fight_pts = v,
                "boss_pts" => p.boss_pts = v,
                "danger" => p.danger = v,
                "tichu_fight" => p.tichu_fight = v,
                "bomb_mode" => p.bomb_mode = v,
                "partner_single" => p.partner_single = v,
                "shed" => p.shed = v,
                "mj_first" => p.mj_first = v,
                "wish_mode" => p.wish_mode = v,
                "overtake" => p.overtake = v,
                "dragon_mode" => p.dragon_mode = v,
                _ => return None,
            }
        }
        Some(p)
    }
}

impl H2Bot {
    pub fn new() -> Self {
        H2Bot { base: HeuristicBot::new(), p: Params::default() }
    }
    pub fn with_params(p: Params) -> Self {
        H2Bot { base: HeuristicBot::new(), p }
    }
}

impl Default for H2Bot {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Rank-level hand shape
// ---------------------------------------------------------------------------

/// A combination described by rank only. `top` is the combo's rank (the
/// triple of a full house, the highest card of a run); `len` its card count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RCombo {
    pub kind: ComboKind,
    pub top: f32,
    pub len: usize,
}

/// Normal cards by rank (index 1 = the Mah Jong, which only goes in runs
/// or alone), plus the specials.
#[derive(Debug, Clone, Copy, Default)]
pub struct Counts {
    pub n: [u8; 15],
    pub phoenix: bool,
    pub dragon: bool,
    pub dog: bool,
}

impl Counts {
    pub fn of(cards: impl IntoIterator<Item = Card>) -> Counts {
        let mut k = Counts::default();
        for c in cards {
            match c {
                Card::Normal { rank, .. } => k.n[rank as usize] += 1,
                Card::Special(SpecialKind::MahJong) => k.n[1] += 1,
                Card::Special(SpecialKind::Phoenix) => k.phoenix = true,
                Card::Special(SpecialKind::Dragon) => k.dragon = true,
                Card::Special(SpecialKind::Dog) => k.dog = true,
            }
        }
        k
    }
}

/// The fewest-combination split of the normal cards (and Mah Jong) found by
/// trying every set of runs, then grouping the rest into bombs, full houses,
/// triples, pairs and singles. The Phoenix, Dragon and Dog are left out:
/// callers count them separately.
#[derive(Debug, Clone, Default)]
pub struct Shape {
    pub combos: Vec<RCombo>,
}

impl Shape {
    pub fn cost(&self) -> usize {
        self.combos.len()
    }
}

fn singles(c: &[RCombo]) -> usize {
    c.iter().filter(|x| x.kind == ComboKind::Single).count()
}

fn sets_of(n: &[u8; 15], out: &mut Vec<RCombo>) {
    let (mut triples, mut pairs) = (vec![], vec![]);
    for r in 1..=14 {
        let top = r as f32;
        match n[r] {
            0 => {}
            1 => out.push(RCombo { kind: ComboKind::Single, top, len: 1 }),
            2 => pairs.push(r),
            3 => triples.push(r),
            _ => out.push(RCombo { kind: ComboKind::FourOfAKindBomb, top, len: 4 }),
        }
    }
    // Pair the lowest triples with the lowest pairs.
    let fh = triples.len().min(pairs.len());
    for (i, &t) in triples.iter().enumerate() {
        if i < fh {
            out.push(RCombo { kind: ComboKind::FullHouse, top: t as f32, len: 5 });
        } else {
            out.push(RCombo { kind: ComboKind::Triple, top: t as f32, len: 3 });
        }
    }
    for &p in &pairs[fh..] {
        out.push(RCombo { kind: ComboKind::Pair, top: p as f32, len: 2 });
    }
}

fn better(a: &[RCombo], b: &[RCombo]) -> bool {
    (a.len(), singles(a)) < (b.len(), singles(b))
}

fn search(n: &mut [u8; 15], from: usize, runs: &mut Vec<RCombo>, best: &mut Vec<RCombo>, budget: &mut usize) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;
    // Option: no more runs.
    let mut all = runs.clone();
    sets_of(n, &mut all);
    if best.is_empty() || better(&all, best) {
        *best = all;
    }
    for r in from..=13 {
        if n[r] == 0 {
            continue;
        }
        // Straights starting at r.
        let mut end = r;
        while end + 1 <= 14 && n[end + 1] > 0 {
            end += 1;
        }
        if end + 1 - r >= 5 {
            for top in (r + 4)..=end {
                for x in r..=top {
                    n[x] -= 1;
                }
                runs.push(RCombo { kind: ComboKind::Straight, top: top as f32, len: top + 1 - r });
                search(n, r, runs, best, budget);
                runs.pop();
                for x in r..=top {
                    n[x] += 1;
                }
            }
        }
        // Consecutive pairs starting at r (not with the Mah Jong).
        if r >= 2 && n[r] >= 2 {
            let mut end = r;
            while end + 1 <= 14 && n[end + 1] >= 2 {
                end += 1;
            }
            for top in (r + 1)..=end {
                for x in r..=top {
                    n[x] -= 2;
                }
                runs.push(RCombo { kind: ComboKind::ConsecutivePairs, top: top as f32, len: 2 * (top + 1 - r) });
                search(n, r, runs, best, budget);
                runs.pop();
                for x in r..=top {
                    n[x] += 2;
                }
            }
        }
    }
}

fn shape_uncached(n: &[u8; 15]) -> Shape {
    let mut n = *n;
    let mut best = vec![];
    let mut budget = 4000;
    search(&mut n, 1, &mut vec![], &mut best, &mut budget);
    Shape { combos: best }
}

thread_local! {
    /// Shapes by packed rank counts. Search bots ask about the same hands
    /// over and over (every rollout replays the same few positions).
    static SHAPES: std::cell::RefCell<std::collections::HashMap<u64, Shape>> = Default::default();
}

pub fn shape(k: &Counts) -> Shape {
    let key = k.n.iter().fold(0u64, |acc, &c| acc * 5 + c as u64);
    SHAPES.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(s) = m.get(&key) {
            return s.clone();
        }
        if m.len() > 1 << 20 {
            m.clear();
        }
        let s = shape_uncached(&k.n);
        m.insert(key, s.clone());
        s
    })
}

/// Combos needed to play out a hand: the shape plus a Dragon and a Dog (the
/// Phoenix is a free agent and isn't counted).
pub fn cost(k: &Counts) -> usize {
    shape(k).cost() + k.dragon as usize + k.dog as usize
}

// ---------------------------------------------------------------------------
// Card tracking
// ---------------------------------------------------------------------------

/// Could some combination in `u` (the unseen cards) beat `c`, bombs aside?
/// An upper bound: the cards may be split between several hands.
pub fn could_beat(c: &RCombo, u: &Counts) -> bool {
    let ph = u.phoenix as u8;
    let above = |lo: usize| (lo.max(2))..=14usize;
    match c.kind {
        ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb => false,
        ComboKind::Single => {
            if c.top >= 15.0 {
                return false;
            }
            u.dragon || u.phoenix || above(c.top.floor() as usize + 1).any(|q| u.n[q] > 0)
        }
        ComboKind::Pair => above(c.top as usize + 1).any(|q| u.n[q] + ph >= 2),
        ComboKind::Triple => above(c.top as usize + 1).any(|q| u.n[q] + ph >= 3),
        ComboKind::FullHouse => above(c.top as usize + 1).any(|q| {
            let need3 = 3u8.saturating_sub(u.n[q]);
            if need3 > ph {
                return false;
            }
            let left = ph - need3;
            (2..=14).any(|p| p != q && u.n[p] + left >= 2)
        }),
        ComboKind::Straight => {
            let len = c.len;
            ((c.top as usize + 1)..=14).any(|t| {
                if t + 1 < len + 1 {
                    return false;
                }
                let lo = t + 1 - len;
                if lo < 1 {
                    return false;
                }
                let missing = (lo..=t).filter(|&x| u.n[x] == 0).count() as u8;
                missing <= ph
            })
        }
        ComboKind::ConsecutivePairs => {
            let k = c.len / 2;
            ((c.top as usize + 1)..=14).any(|t| {
                if t < k + 1 {
                    return false;
                }
                let lo = t + 1 - k;
                let deficit: u8 = (lo..=t).map(|x| 2u8.saturating_sub(u.n[x])).sum();
                deficit <= ph
            })
        }
    }
}

fn rcombo(c: &Combo) -> RCombo {
    RCombo { kind: c.kind, top: c.rank, len: c.length }
}

// ---------------------------------------------------------------------------
// Situation
// ---------------------------------------------------------------------------

struct Situation {
    unseen: Counts,
    /// An opponent (not out) who called Tichu while nobody is out yet.
    live_opp_tichu: [bool; 4],
    partner_live_tichu: bool,
    my_live_tichu: bool,
    /// An opponent went out first: the other one is chasing a 1-2.
    opp_chasing_12: bool,
    /// My partner went out first: I'm chasing a 1-2.
    we_chase_12: bool,
    /// Fewest cards held by an opponent still in (99 if none).
    opp_min: usize,
}

fn situation(view: &PlayerView) -> Situation {
    let nobody_out = view.players.iter().all(|p| !p.is_out());
    let first_out = (0..4).find(|&s| view.players[s].out_order == 1);
    let opps = view.opponents();
    let mut live_opp_tichu = [false; 4];
    for &s in &opps {
        live_opp_tichu[s] = nobody_out && view.players[s].tichu != TichuCall::None;
    }
    let partner = view.partner();
    let opp_min =
        opps.iter().filter(|&&s| !view.players[s].is_out()).map(|&s| view.players[s].card_count).min().unwrap_or(99);
    Situation {
        unseen: Counts::of(view.unseen().cards()),
        live_opp_tichu,
        partner_live_tichu: nobody_out && view.players[partner].tichu != TichuCall::None,
        my_live_tichu: nobody_out && view.me().tichu != TichuCall::None,
        opp_chasing_12: first_out.is_some_and(|s| !view.is_teammate(s))
            && opps.iter().any(|&s| !view.players[s].is_out()),
        we_chase_12: first_out == Some(partner),
        opp_min,
    }
}

/// How a hand plays out from here.
struct Plan {
    /// Combinations that win when led: bosses of my split, the Dragon, bombs.
    lead_bosses: usize,
    /// Ways to win back the lead when following: the Dragon, a loose
    /// Phoenix, bombs, boss singles.
    regain: usize,
    /// The other combinations, the Dog included.
    losers: usize,
}

fn plan(mine: &Counts, unseen: &Counts, base_cost: usize) -> Plan {
    let sh = shape(mine);
    let bosses: Vec<&RCombo> = sh.combos.iter().filter(|c| !could_beat(c, unseen)).collect();
    let regain = bosses.iter().filter(|c| c.kind == ComboKind::Single || c.kind == ComboKind::FourOfAKindBomb).count()
        + mine.dragon as usize
        + mine.phoenix as usize;
    Plan {
        lead_bosses: bosses.len() + mine.dragon as usize,
        regain,
        losers: base_cost - bosses.len() - mine.dragon as usize,
    }
}

fn minus(hand: &[Card], play: &Combo) -> Counts {
    let used = play.card_set();
    Counts::of(hand.iter().copied().filter(|c| !used.contains(*c)))
}

/// Combos still needed after `play` (the Phoenix never counts).
fn cost_after(hand: &[Card], play: &Combo) -> usize {
    cost(&minus(hand, play))
}

impl Bot for H2Bot {
    fn name(&self) -> String {
        "h2".into()
    }

    fn grand_tichu(&mut self, view: &PlayerView) -> bool {
        calls::grand_tichu_v2(view)
    }

    fn small_tichu(&mut self, view: &PlayerView) -> bool {
        calls::small_tichu_v2(view, calls::DEFAULT_TICHU_THRESHOLD).unwrap_or(false)
    }

    fn pass_cards(&mut self, view: &PlayerView) -> PassInfo {
        self.base.pass_cards(view)
    }

    fn play(&mut self, view: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let hand = &view.hand;
        if let Some(i) = plays.iter().position(|c| c.cards.len() == hand.len()) {
            return Some(i);
        }
        let sit = situation(view);
        let mine = Counts::of(hand.iter().copied());
        let base_cost = cost(&mine);
        if view.is_leading() {
            Some(self.lead(view, plays, &sit, &mine, base_cost))
        } else {
            self.follow(view, plays, can_pass, &sit, base_cost)
        }
    }

    fn bomb(&mut self, view: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        if self.p.bomb_mode == 0.0 {
            return self.base.bomb(view, bombs);
        }
        let by = view.last_played_by?;
        if view.is_teammate(by) {
            return None;
        }
        let sit = situation(view);
        let cheapest = || {
            (0..bombs.len()).min_by(|&a, &b| {
                let ka = (bombs[a].length, bombs[a].rank);
                let kb = (bombs[b].length, bombs[b].rank);
                ka.partial_cmp(&kb).unwrap()
            })
        };
        // A bomb that leaves me one combination from going out: take the lead.
        if bombs.iter().any(|b| cost_after(&view.hand, b) <= 1 && !view.hand.contains(&PHOENIX)) {
            return (0..bombs.len()).find(|&i| cost_after(&view.hand, &bombs[i]) <= 1);
        }
        if view.trick_countdown.is_none() {
            return None;
        }
        let opp = &view.players[by];
        let near_out = !opp.is_out() && opp.card_count <= 2;
        let caller = sit.live_opp_tichu[by] && opp.card_count <= 6;
        let chasing = sit.opp_chasing_12 && !opp.is_out() && opp.card_count <= 5;
        if near_out || caller || chasing || view.trick_points() >= 20 {
            return cheapest();
        }
        None
    }

    fn wish(&mut self, view: &PlayerView) -> Option<Rank> {
        if self.p.wish_mode == 0.0 {
            return self.base.wish(view);
        }
        // The next player (always an opponent) called: make them give up an Ace.
        let next = view.next_seat(view.seat);
        if view.players[next].tichu != TichuCall::None && !view.players[next].is_out() {
            return Some(14);
        }
        // Otherwise the card I passed the next player: they must play it on
        // the Mah Jong right away, so the wish never binds me. Failing that
        // (they've played it already), the one I passed the other opponent.
        let held = view.known_holdings();
        let passed_to = |s: Seat| held.iter().find(|(_, t)| *t == s).and_then(|(c, _)| c.rank());
        let [a, b] = view.opponents();
        let other = if a == next { b } else { a };
        passed_to(next).or(passed_to(other))
    }

    fn dragon_recipient(&mut self, view: &PlayerView) -> Seat {
        if self.p.dragon_mode == 0.0 {
            return self.base.dragon_recipient(view);
        }
        let [a, b] = view.opponents();
        let key = |s: Seat| (view.players[s].tichu == TichuCall::None, view.players[s].card_count);
        if key(a) >= key(b) { a } else { b }
    }
}

impl H2Bot {
    fn lead(&self, view: &PlayerView, plays: &[Combo], sit: &Situation, mine: &Counts, base_cost: usize) -> usize {
        let hand = &view.hand;
        let partner = view.partner();
        let partner_in = !view.players[partner].is_out();
        let partner_cards = view.players[partner].card_count;

        // Lead the Mah Jong first, to get the wish in early.
        if self.p.mj_first > 0.0 && mine.n[1] > 0 {
            let mj: Vec<usize> = (0..plays.len()).filter(|&i| plays[i].cards.contains(&MAH_JONG)).collect();
            if !mj.is_empty() {
                // The one that leaves the rest of the hand in the best shape.
                return *mj.iter().min_by_key(|&&i| (cost_after(hand, &plays[i]), std::cmp::Reverse(plays[i].cards.len()))).unwrap();
            }
        }

        // Hand the lead to a partner who's going for it.
        if let Some(i) = plays.iter().position(|c| c.cards == [DOG]) {
            if partner_in && (sit.partner_live_tichu || partner_cards < hand.len()) && !sit.my_live_tichu {
                return i;
            }
        }

        let p = plan(mine, &sit.unseen, base_cost);

        // Score each play. Clean plays (whole combinations of the best split)
        // come first; then bosses or losers by the plan.
        let cash_out = p.losers <= 1 && p.lead_bosses > 0;
        let danger = sit.opp_min; // an opponent holding this few cards can only follow short combos
        let score = |c: &Combo| -> f32 {
            let after = cost_after(hand, c);
            let clean = after + 1 <= base_cost;
            let boss = c.cards == [DRAGON] || !could_beat(&rcombo(c), &sit.unseen);
            let mut s = 0.0;
            s -= 10.0 * (after as f32 - (base_cost as f32 - 1.0)).max(0.0);
            if !clean {
                s -= 5.0;
            }
            if c.is_bomb() {
                s -= if cash_out { 2.0 } else { 30.0 };
            }
            if c.cards == [DOG] {
                s -= 8.0;
            }
            if c.cards == [DRAGON] {
                // The Dragon's trick goes to an opponent: lead it last.
                s -= 6.0;
            }
            if c.contains(SpecialKind::Phoenix) {
                s -= 8.0;
            }
            if cash_out {
                if boss {
                    s += 20.0;
                }
            } else {
                if boss {
                    s -= 12.0;
                }
                // Shed cards, lowest first.
                s += c.cards.len() as f32 * self.p.shed - c.rank * 0.6;
                if sit.partner_live_tichu && c.kind == ComboKind::Single {
                    s += self.p.partner_single;
                }
            }
            // Don't lead what a nearly-out opponent can follow.
            if danger as f32 <= self.p.danger && c.cards.len() <= danger && !boss {
                s -= 25.0;
                if c.kind == ComboKind::Single {
                    s += c.rank; // if we must, lead high
                }
            }
            // A partner on one or two cards: give them something to follow.
            if partner_in && partner_cards <= 2 && c.cards.len() == partner_cards && danger > partner_cards {
                s += 6.0 - c.rank * 0.3;
            }
            s
        };
        best_by(plays, score).unwrap()
    }

    fn follow(
        &self,
        view: &PlayerView,
        plays: &[Combo],
        can_pass: bool,
        sit: &Situation,
        base_cost: usize,
    ) -> Option<usize> {
        let hand = &view.hand;
        let partner = view.partner();
        let by = view.last_played_by.unwrap();
        let partner_winning = by == partner;
        if partner_winning && can_pass {
            return self.overtake(view, plays, sit, base_cost);
        }
        let points = view.trick_points();
        let opp = &view.players[by];
        let winner_threat = !partner_winning
            && (sit.live_opp_tichu[by] || opp.card_count <= 3 || (sit.opp_chasing_12 && !opp.is_out()));
        let points_f = points as f32;
        let fight = winner_threat
            || points_f >= self.p.fight_pts
            || sit.we_chase_12
            || (sit.my_live_tichu && self.p.tichu_fight > 0.0)
            || hand.len() <= 4;
        let hard_fight = !partner_winning
            && ((opp.card_count <= 2 && !opp.is_out())
                || (sit.opp_chasing_12 && opp.card_count <= 5 && !opp.is_out())
                || (sit.live_opp_tichu[by] && opp.card_count <= 6)
                || points >= 25);

        let mine = Counts::of(hand.iter().copied());
        let p = plan(&mine, &sit.unseen, base_cost);
        // I can run out from the lead: spend controls freely.
        let running = p.losers <= p.regain + 1;

        let mut best: Option<(f32, usize)> = None;
        for (i, c) in plays.iter().enumerate() {
            let after = cost_after(hand, c);
            let damage = after as f32 - (base_cost as f32 - 1.0);
            let boss = c.cards == [DRAGON] || !could_beat(&rcombo(c), &sit.unseen);
            let allowed = if c.is_bomb() {
                hard_fight || (after <= 1 && !sit.opp_chasing_12) || !can_pass
            } else if c.cards == [DRAGON] {
                fight || running || !can_pass
            } else if c.cards == [PHOENIX] {
                (boss && (points_f >= self.p.boss_pts || running)) || fight || !can_pass
            } else if boss {
                fight || points_f >= self.p.boss_pts || running || !can_pass
            } else if damage > 0.0 {
                fight || !can_pass
            } else {
                true
            };
            if !allowed {
                continue;
            }
            let mut s = -c.rank - 6.0 * damage.max(0.0);
            if c.is_bomb() {
                s -= 40.0;
            }
            if c.contains(SpecialKind::Phoenix) {
                s -= if c.length > 1 { 4.0 } else { 8.0 };
            }
            if c.cards == [DRAGON] {
                s -= 10.0;
            }
            if best.is_none_or(|(b, _)| s > b) {
                best = Some((s, i));
            }
        }
        match best {
            Some((_, i)) => Some(i),
            None if can_pass => None,
            None => Some(0),
        }
    }
}

impl H2Bot {
    /// My partner is winning the trick. Usually pass, but take over a low
    /// single or pair to shed a card cheaply (unless they called), and as the
    /// Tichu caller take over to shed any loser or to run out.
    fn overtake(&self, view: &PlayerView, plays: &[Combo], sit: &Situation, base_cost: usize) -> Option<usize> {
        if self.p.overtake == 0.0 {
            return None;
        }
        let trick = view.current_trick.as_ref()?;
        let hand = &view.hand;
        let partner_called = view.players[view.partner()].tichu != TichuCall::None;
        let p = plan(&Counts::of(hand.iter().copied()), &sit.unseen, base_cost);
        let running = p.losers <= p.regain + 1;
        let low = (trick.kind == ComboKind::Single && trick.rank < 10.0)
            || (trick.kind == ComboKind::Pair && trick.rank < 8.0);
        let mut best: Option<(f32, usize)> = None;
        for (i, c) in plays.iter().enumerate() {
            if c.is_bomb() || c.contains(SpecialKind::Dragon) || c.contains(SpecialKind::Phoenix) {
                continue;
            }
            let damage = cost_after(hand, c) as f32 - (base_cost as f32 - 1.0);
            let boss = !could_beat(&rcombo(c), &sit.unseen);
            let ok = if sit.my_live_tichu {
                running || (!boss && damage <= 0.0)
            } else {
                low && !partner_called && !boss && damage <= 0.0
            };
            if !ok {
                continue;
            }
            let s = -c.rank - 6.0 * damage.max(0.0);
            if best.is_none_or(|(b, _)| s > b) {
                best = Some((s, i));
            }
        }
        best.map(|(_, i)| i)
    }
}

fn best_by(plays: &[Combo], score: impl Fn(&Combo) -> f32) -> Option<usize> {
    (0..plays.len()).max_by(|&a, &b| score(&plays[a]).partial_cmp(&score(&plays[b])).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: Suit, r: Rank) -> Card {
        Card::normal(s, r)
    }

    #[test]
    fn shape_finds_runs_and_sets() {
        use Suit::*;
        // 3-7 straight + pair of 9s + K single.
        let h = [n(Jade, 3), n(Sword, 4), n(Jade, 5), n(Star, 6), n(Pagoda, 7), n(Jade, 9), n(Star, 9), n(Jade, 13)];
        let s = shape(&Counts::of(h));
        assert_eq!(s.cost(), 3);
        // 55 66 77 stairs + triple 10s + pair Js = stairs + full house.
        let h = [
            n(Jade, 5),
            n(Star, 5),
            n(Jade, 6),
            n(Star, 6),
            n(Jade, 7),
            n(Star, 7),
            n(Jade, 10),
            n(Star, 10),
            n(Sword, 10),
            n(Jade, 11),
            n(Star, 11),
        ];
        assert_eq!(shape(&Counts::of(h)).cost(), 2);
    }

    #[test]
    fn bosses() {
        use Suit::*;
        let mut u = Counts::of([n(Jade, 13), n(Star, 13), n(Sword, 9)]);
        let ace = RCombo { kind: ComboKind::Single, top: 14.0, len: 1 };
        let pair_q = RCombo { kind: ComboKind::Pair, top: 12.0, len: 2 };
        let pair_k = RCombo { kind: ComboKind::Pair, top: 13.0, len: 2 };
        assert!(!could_beat(&ace, &u));
        assert!(could_beat(&pair_q, &u));
        assert!(!could_beat(&pair_k, &u));
        u.phoenix = true;
        assert!(could_beat(&ace, &u));
    }
}
