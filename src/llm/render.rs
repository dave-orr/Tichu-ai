//! Text for a language-model player: the rules (system prompt), a rendering
//! of one `PlayerView`, numbered option lists for each kind of decision, and
//! parsing of the answer.
//!
//! Everything here is built from the `PlayerView` alone. Seats are named
//! relative to the player ("You", "Partner", "Left opp", "Right opp").

use std::fmt::Write;

use crate::engine::Action;
use crate::scoring::total_points;
use crate::types::*;
use crate::view::PlayerView;

/// The fixed system prompt: rules, notation and answer format.
pub const SYSTEM_PROMPT: &str = r#"You are an expert Tichu player. You are one seat of a four-player game, partnered with the player opposite you, and you will be asked for one decision at a time. Each request is self-contained: it describes everything you can see.

GOAL. Each deal is scored on its own, starting from 0-0: maximize your team's points minus the opponents' points for this round. Long-run game score doesn't matter.

RULES (the server's rules; follow these, not other variants).
- 56 cards: four suits (Jade, Sword, Pagoda, Star) of 2..10, J, Q, K, A, plus four specials: Mah Jong (the 1), Dog, Phoenix, Dragon. 14 cards each. Before play, each player passes one card to each other player.
- Points: 5s = 5, 10s and Kings = 10, Dragon = +25, Phoenix = -25, everything else 0. 100 points in total.
- Combinations: single; pair; triple; full house (triple + pair, ranked by the triple); straight (5+ consecutive ranks, any suits, A high, no wrap); stairs (2+ consecutive pairs, e.g. 5566). To beat a play you must play the same type, the same length, and a higher rank. Passing doesn't stop you playing later in the same trick. A trick ends when everyone else still in passes after the last play; that player wins the cards and leads next.
- Bombs: four of a kind, or a straight flush of 5+ cards. A bomb beats any non-bomb and can be thrown at any time, even out of turn. Straight flushes beat four-of-a-kinds; longer straight flushes beat shorter; otherwise the higher rank wins.
- Mah Jong: rank 1, can start a straight (1-2-3-4-5). Its holder leads the first trick. Whoever plays it may wish for a rank 2..A; while the wish is open, a player who can make a legal play containing the wished rank must do so. The wish ends when that rank is played.
- Dog: only as a lead, as a single. It passes the lead to your partner (or the next player still in, if your partner is out). Worth nothing; can't be beaten or bombed.
- Phoenix: wild in pairs, triples, full houses, straights and stairs (never in bombs). As a single it counts half a rank above the card it's played on (1.5 when led); it beats an Ace but not the Dragon.
- Dragon: the highest single, beaten only by a bomb. If the Dragon wins the trick, its player must give that whole trick to one of the two opponents.
- The round ends when three players are out. The last player gives the cards left in their hand to the opponents, and the tricks they won to the player who went out first.
- 1-2 finish: if partners go out first and second, their team scores 200 and nobody counts card points.
- Tichu calls: Small Tichu (before your first card) is +100 if the caller goes out first, -100 if not; Grand Tichu (after the first 8 cards) is +-200. Calls are decided for you by a fixed policy; you only make playing decisions, but play to make your team's calls and break the opponents'.

NOTATION. A card is rank then suit: 2..9, T = 10, J, Q, K, A; suits j = Jade, s = Sword, p = Pagoda, * = Star. Specials: MJ (Mah Jong), Dog, Ph (Phoenix), Dr (Dragon). Play order is You -> Right opp -> Partner -> Left opp -> You.

ANSWER FORMAT. Think as much as you need, then end your reply with one final line of the form
ANSWER: <your choice>
exactly as the request specifies. Nothing after that line."#;

/// Seat name relative to `view.seat`.
pub fn who(view: &PlayerView, seat: Seat) -> &'static str {
    let me = view.seat;
    if seat == me {
        "You"
    } else if seat == partner_seat(me) {
        "Partner"
    } else if seat == right_seat(me) {
        "Right opp"
    } else {
        "Left opp"
    }
}

fn sorted(cards: &[Card]) -> Vec<Card> {
    let mut v = cards.to_vec();
    v.sort_by_key(|c| c.sort_key());
    v
}

pub fn cards(cs: &[Card]) -> String {
    if cs.is_empty() {
        return "(none)".into();
    }
    cards_str(&sorted(cs))
}

fn rank_name(r: f32) -> String {
    match r {
        x if x == 1.0 => "1".into(),
        x if x == 15.0 => "Dragon".into(),
        x if x.fract() != 0.0 => format!("{x}"),
        x => match x as u8 {
            10 => "10".into(),
            11 => "J".into(),
            12 => "Q".into(),
            13 => "K".into(),
            14 => "A".into(),
            n => n.to_string(),
        },
    }
}

/// e.g. `Pair of 7s: 7j 7s`, `Straight, 6 cards, to K: 8j 9s Tp J* Qj Ph (Ph = K)`.
pub fn describe(c: &Combo) -> String {
    let r = rank_name(c.rank);
    let what = match c.kind {
        ComboKind::Single if c.cards == [DOG] => "Dog".to_string(),
        ComboKind::Single if c.cards == [PHOENIX] => format!("Single Phoenix ({r})"),
        ComboKind::Single => "Single".to_string(),
        ComboKind::Pair => format!("Pair of {r}s"),
        ComboKind::Triple => format!("Triple {r}s"),
        ComboKind::FullHouse => format!("Full house, {r}s full"),
        ComboKind::Straight => format!("Straight, {} cards, to {r}", c.length),
        ComboKind::ConsecutivePairs => format!("Stairs, {} pairs, to {r}", c.length / 2),
        ComboKind::FourOfAKindBomb => format!("BOMB: four {r}s"),
        ComboKind::StraightFlushBomb => format!("BOMB: straight flush, {} cards, to {r}", c.length),
    };
    let mut s = format!("{what}: {}", cards(&c.cards));
    if let (Some(p), true) = (c.phoenix_as, c.cards.len() > 1) {
        s += &format!(" (Ph = {})", rank_name(p as f32));
    }
    s
}

/// The situation, as text: scores, every player's public state, my hand,
/// the pass, the table, the round so far, and the unseen cards.
pub fn situation(view: &PlayerView) -> String {
    let mut s = String::new();
    let me = view.seat;
    let order: Vec<Seat> = {
        let mut v = vec![me];
        while v.len() < 4 {
            v.push(view.next_seat(*v.last().unwrap()));
        }
        v
    };

    let _ = writeln!(s, "PLAYERS (in play order)");
    for &p in &order {
        let pl = &view.players[p];
        let mut line = format!("- {:<9}", who(view, p));
        if pl.is_out() {
            line += &format!(" OUT ({})", ordinal(pl.out_order));
        } else {
            line += &format!(" {:>2} cards", pl.card_count);
        }
        match pl.tichu {
            TichuCall::Small => line += " | called TICHU",
            TichuCall::Grand => line += " | called GRAND TICHU",
            TichuCall::None => {}
        }
        if pl.tichu == TichuCall::None && !pl.has_played_first_card && view.phase == Phase::Playing {
            line += " | hasn't played yet";
        }
        line += &format!(" | won {} tricks, {} pts", pl.trick_count, pl.captured_points);
        let _ = writeln!(s, "{line}");
    }

    let _ = writeln!(s, "\nYOUR HAND ({} cards): {}", view.hand.len(), cards(&view.hand));

    if let Some(pass) = view.my_pass {
        let _ = writeln!(
            s,
            "You passed: {} to Left opp, {} to Partner, {} to Right opp.",
            pass.left, pass.partner, pass.right
        );
    }
    if !view.received.is_empty() {
        let got: Vec<String> = view.received.iter().map(|(c, from)| format!("{c} from {}", who(view, *from))).collect();
        let _ = writeln!(s, "You received: {}.", got.join(", "));
    }

    if view.phase == Phase::Playing {
        let _ = writeln!(s, "\nROUND SO FAR");
        let _ = write!(s, "{}", history(view));

        let _ = writeln!(s, "\nTABLE");
        match &view.current_trick {
            None => {
                let _ = writeln!(s, "No trick in progress.");
            }
            Some(top) => {
                for tp in &view.trick_plays {
                    let _ = writeln!(s, "- {}: {}", who(view, tp.seat), describe(&tp.combo));
                }
                let by = view.last_played_by.map(|p| who(view, p)).unwrap_or("?");
                let _ = writeln!(
                    s,
                    "To beat: {} (played by {by}). Points on the table: {}.",
                    describe(top),
                    view.trick_points()
                );
                if view.pass_count > 0 {
                    let _ = writeln!(s, "Passes since that play: {}.", view.pass_count);
                }
            }
        }
        match view.wish {
            Some(w) => {
                let _ = writeln!(s, "Open Mah Jong wish: {}.", rank_name(w as f32));
            }
            None => {}
        }

        let _ = writeln!(s, "\nCARD COUNTING");
        let _ = writeln!(s, "Unseen (in the other three hands): {}", cards(&view.unseen().cards()));
        let known = view.known_holdings();
        if !known.is_empty() {
            let k: Vec<String> = known.iter().map(|(c, p)| format!("{c} ({})", who(view, *p))).collect();
            let _ = writeln!(s, "Known locations (cards you passed, not yet played): {}", k.join(", "));
        }
    }
    s
}

fn ordinal(n: u8) -> &'static str {
    match n {
        1 => "1st",
        2 => "2nd",
        3 => "3rd",
        _ => "4th",
    }
}

/// This round's plays, one line per trick.
fn history(view: &PlayerView) -> String {
    let mut out = String::new();
    let mut line: Vec<String> = Vec::new();
    let mut trick_cards: Vec<Card> = Vec::new();
    let mut last: Option<Seat> = None;
    let mut n = 0;
    let flush = |out: &mut String, line: &mut Vec<String>, end: String, n: &mut usize| {
        if !line.is_empty() {
            *n += 1;
            let _ = writeln!(out, "{}. {}{end}", n, line.join(", "));
            line.clear();
        }
    };
    for a in &view.log {
        match a {
            Action::SmallTichu { seat } => line.push(format!("[{} calls Tichu]", who(view, *seat))),
            Action::Play { seat, cards: cs, phoenix_as } => {
                let mut p = format!("{} {}", who(view, *seat), cards(cs));
                if let (Some(r), true) = (phoenix_as, cs.len() > 1) {
                    p += &format!(" (Ph={})", rank_name(*r as f32));
                }
                line.push(p);
                if cs == &[DOG] {
                    flush(&mut out, &mut line, " -> Dog: lead to partner".into(), &mut n);
                    continue;
                }
                trick_cards.extend_from_slice(cs);
                last = Some(*seat);
            }
            Action::Bomb { seat, cards: cs } => {
                line.push(format!("{} BOMBS {}", who(view, *seat), cards(cs)));
                trick_cards.extend_from_slice(cs);
                last = Some(*seat);
            }
            Action::PassTurn { seat } => line.push(format!("{} pass", who(view, *seat))),
            Action::Wish { seat, rank } => match rank {
                Some(r) => line.push(format!("[{} wishes {}]", who(view, *seat), rank_name(*r as f32))),
                None => line.push(format!("[{} makes no wish]", who(view, *seat))),
            },
            Action::ResolveTrick => {
                if trick_cards.contains(&DRAGON) {
                    continue; // finished by GiveDragon
                }
                let w = last.map(|p| who(view, p)).unwrap_or("?");
                let end = format!(" -> {w} wins {} pts", total_points(&trick_cards));
                flush(&mut out, &mut line, end, &mut n);
                trick_cards.clear();
            }
            Action::GiveDragon { seat, to } => {
                let end = format!(
                    " -> {} wins with the Dragon, gives {} pts to {}",
                    who(view, *seat),
                    total_points(&trick_cards),
                    who(view, *to)
                );
                flush(&mut out, &mut line, end, &mut n);
                trick_cards.clear();
            }
            Action::GrandTichu { .. } | Action::PassCards { .. } => {}
        }
    }
    flush(&mut out, &mut line, " (in progress)".into(), &mut n);
    if n == 0 {
        out += "(nothing played yet)\n";
    }
    out
}

/// A decision put to the model: the prompt and what a valid answer is.
#[derive(Debug, Clone)]
pub struct Question {
    pub kind: &'static str,
    pub prompt: String,
}

fn ask(view: &PlayerView, task: &str) -> String {
    format!("{}\nDECISION\n{task}", situation(view))
}

/// Indices into `plays` to offer, in menu order (grouped by type, low to
/// high), so the model sees a tidy menu while answers map back exactly.
///
/// Plays that differ only in suits (`Tj Ts` vs `Tj T*`) are offered once:
/// suits only matter for straight-flush bombs, so the variant kept is the
/// one that leaves the most bombs in `hand` (then the engine's first).
pub fn menu_order(plays: &[Combo], hand: &[Card]) -> Vec<usize> {
    let kind_order = |k: ComboKind| match k {
        ComboKind::Single => 0,
        ComboKind::Pair => 1,
        ComboKind::Triple => 2,
        ComboKind::FullHouse => 3,
        ComboKind::Straight => 4,
        ComboKind::ConsecutivePairs => 5,
        ComboKind::FourOfAKindBomb => 6,
        ComboKind::StraightFlushBomb => 7,
    };
    // Same type, length, rank, Phoenix reading, and card ranks.
    let shape = |c: &Combo| {
        let mut ranks: Vec<u8> = c.cards.iter().map(|x| x.sort_key().0).collect();
        ranks.sort_unstable();
        (kind_order(c.kind), c.length, c.rank.to_bits(), c.phoenix_as, ranks)
    };
    let bombs_left = |c: &Combo| {
        let mut rest = CardSet::from_cards(hand);
        for x in &c.cards {
            rest.remove(*x);
        }
        crate::combinations::playable_combos(&rest.cards(), None).iter().filter(|b| b.is_bomb()).count()
    };
    let mut best: std::collections::HashMap<_, (usize, usize)> = std::collections::HashMap::new();
    for (i, c) in plays.iter().enumerate() {
        let key = shape(c);
        let b = if c.is_bomb() { 0 } else { bombs_left(c) };
        match best.get(&key) {
            Some(&(_, kept)) if kept >= b => {}
            _ => {
                best.insert(key, (i, b));
            }
        }
    }
    let mut idx: Vec<usize> = best.into_values().map(|(i, _)| i).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (&plays[a], &plays[b]);
        (kind_order(x.kind), x.length, x.rank)
            .partial_cmp(&(kind_order(y.kind), y.length, y.rank))
            .unwrap()
            .then(a.cmp(&b))
    });
    idx
}

/// On my turn. Options are numbered from 1; 0 is pass (when allowed).
pub fn play_question(view: &PlayerView, plays: &[Combo], can_pass: bool) -> (Question, Vec<usize>) {
    let order = menu_order(plays, &view.hand);
    let mut t = String::new();
    if view.current_trick.is_none() {
        t += "It's your turn to lead. Choose a play:\n";
    } else {
        t += "It's your turn. Choose a play:\n";
    }
    if can_pass {
        t += "0: Pass\n";
    }
    for (n, &i) in order.iter().enumerate() {
        let _ = writeln!(t, "{}: {}", n + 1, describe(&plays[i]));
    }
    let lo = if can_pass { 0 } else { 1 };
    let _ = write!(t, "\nReply with ANSWER: <number> ({lo}-{}).", order.len());
    (Question { kind: "play", prompt: ask(view, &t) }, order)
}

/// Out-of-turn bomb offer. 0 = don't bomb, 1.. = bombs.
pub fn bomb_question(view: &PlayerView, bombs: &[Combo]) -> Question {
    let mut t = String::new();
    t += "It is NOT your turn, but you may throw a bomb now (out of turn), or let play continue.\n";
    t += "0: Don't bomb\n";
    for (n, b) in bombs.iter().enumerate() {
        let _ = writeln!(t, "{}: {}", n + 1, describe(b));
    }
    let _ = write!(t, "\nReply with ANSWER: <number> (0-{}).", bombs.len());
    Question { kind: "bomb", prompt: ask(view, &t) }
}

/// The Mah Jong wish. Answer 0 (no wish) or a rank 2..14.
pub fn wish_question(view: &PlayerView) -> Question {
    let t = "You just played the Mah Jong. Name a wished rank, or none.\n\
             Reply with ANSWER: <rank>, where rank is 2-14 (11 = J, 12 = Q, 13 = K, 14 = A), or ANSWER: 0 for no wish.";
    Question { kind: "wish", prompt: ask(view, t) }
}

/// Dragon trick recipient. 1 = Right opp, 2 = Left opp.
pub fn dragon_question(view: &PlayerView) -> Question {
    let t = "Your Dragon won the trick. Give the trick to an opponent:\n1: Right opp\n2: Left opp\n\n\
             Reply with ANSWER: 1 or ANSWER: 2.";
    Question { kind: "dragon", prompt: ask(view, t) }
}

/// Card passing. The hand is numbered; answer three numbers: left, partner, right.
pub fn pass_question(view: &PlayerView) -> (Question, Vec<Card>) {
    let hand = sorted(&view.hand);
    let mut t = String::new();
    t += "Pass one card to each other player (passes are simultaneous). Your cards:\n";
    for (n, c) in hand.iter().enumerate() {
        let _ = writeln!(t, "{}: {c}", n + 1);
    }
    let mut calls = Vec::new();
    for p in 0..4 {
        match view.players[p].tichu {
            TichuCall::Grand => calls.push(format!("{} called Grand Tichu", who(view, p))),
            TichuCall::Small => calls.push(format!("{} called Tichu", who(view, p))),
            TichuCall::None => {}
        }
    }
    if !calls.is_empty() {
        let _ = writeln!(t, "Calls so far: {}.", calls.join(", "));
    }
    let _ = write!(
        t,
        "\nReply with ANSWER: <to Left opp> <to Partner> <to Right opp>, three different numbers from 1-{}, e.g. ANSWER: 3 12 1.",
        hand.len()
    );
    (Question { kind: "pass", prompt: format!("{}\nDECISION\n{t}", situation(view)) }, hand)
}

/// The numbers on the last `ANSWER:` line of a reply.
pub fn parse_answer(reply: &str) -> Option<Vec<i64>> {
    let line = reply.lines().rev().find(|l| l.to_ascii_uppercase().contains("ANSWER"))?;
    let up = line.to_ascii_uppercase();
    let at = up.rfind("ANSWER")? + "ANSWER".len();
    let rest = &line[at..];
    let nums: Vec<i64> = rest
        .split(|c: char| !(c.is_ascii_digit() || c == '-'))
        .filter(|t| !t.is_empty() && *t != "-")
        .filter_map(|t| t.parse().ok())
        .collect();
    if nums.is_empty() {
        None
    } else {
        Some(nums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_last_answer_line() {
        assert_eq!(parse_answer("I think 3.\nANSWER: 4"), Some(vec![4]));
        assert_eq!(parse_answer("answer: 2\nmore thought\n**ANSWER:** 7"), Some(vec![7]));
        assert_eq!(parse_answer("ANSWER: 3 12 1"), Some(vec![3, 12, 1]));
        assert_eq!(parse_answer("ANSWER: 3, 12, 1."), Some(vec![3, 12, 1]));
        assert_eq!(parse_answer("no idea"), None);
        assert_eq!(parse_answer("ANSWER: pass"), None);
    }

    #[test]
    fn menu_merges_suit_variants_but_keeps_bombs() {
        use crate::combinations::playable_combos;
        let n = Card::normal;
        // Four 9s and a 10: one pair of 9s and one single 9 on the menu.
        let hand = vec![n(Suit::Jade, 9), n(Suit::Sword, 9), n(Suit::Pagoda, 9), n(Suit::Star, 9), n(Suit::Sword, 10)];
        let plays = playable_combos(&hand, None);
        let menu = menu_order(&plays, &hand);
        let pairs = menu.iter().filter(|&&i| plays[i].kind == ComboKind::Pair).count();
        let singles = menu.iter().filter(|&&i| plays[i].kind == ComboKind::Single).count();
        assert_eq!((pairs, singles), (1, 2));

        // 5-9 of Jade is a straight-flush bomb; a spare 7s. Playing a single 7
        // should use the 7s, not the 7j that the bomb needs.
        let mut hand: Vec<Card> = (5..=9).map(|r| n(Suit::Jade, r)).collect();
        hand.push(n(Suit::Sword, 7));
        let plays = playable_combos(&hand, None);
        let menu = menu_order(&plays, &hand);
        let sevens: Vec<&Combo> =
            menu.iter().map(|&i| &plays[i]).filter(|c| c.kind == ComboKind::Single && c.rank == 7.0).collect();
        assert_eq!(sevens.len(), 1);
        assert_eq!(sevens[0].cards, vec![n(Suit::Sword, 7)]);
    }

    #[test]
    fn describes_combos() {
        let c = crate::combinations::identify_combo(&[Card::normal(Suit::Jade, 7), Card::normal(Suit::Star, 7)], None)
            .unwrap();
        assert_eq!(describe(&c), "Pair of 7s: 7j 7*");
    }
}
