use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Suits & Ranks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Suit {
    Jade,
    Sword,
    Pagoda,
    Star,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Jade, Suit::Sword, Suit::Pagoda, Suit::Star];
}

/// Normal ranks 2..=14  (J=11, Q=12, K=13, A=14).
pub type Rank = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialKind {
    MahJong,
    Dog,
    Phoenix,
    Dragon,
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Card {
    Normal { suit: Suit, rank: Rank },
    Special(SpecialKind),
}

impl Card {
    pub fn is_special(&self) -> bool {
        matches!(self, Card::Special(_))
    }

    pub fn rank(&self) -> Option<Rank> {
        match self {
            Card::Normal { rank, .. } => Some(*rank),
            _ => None,
        }
    }

    pub fn suit(&self) -> Option<Suit> {
        match self {
            Card::Normal { suit, .. } => Some(*suit),
            _ => None,
        }
    }

    /// Unique identifier string (matches TS `cardId`).
    pub fn id(&self) -> String {
        match self {
            Card::Normal { suit, rank } => {
                let s = match suit {
                    Suit::Jade => "jade",
                    Suit::Sword => "sword",
                    Suit::Pagoda => "pagoda",
                    Suit::Star => "star",
                };
                format!("{}-{}", s, rank)
            }
            Card::Special(kind) => match kind {
                SpecialKind::MahJong => "mahjong".into(),
                SpecialKind::Dog => "dog".into(),
                SpecialKind::Phoenix => "phoenix".into(),
                SpecialKind::Dragon => "dragon".into(),
            },
        }
    }

    /// Sort key used for ordering hands (matches TS `sortHand`).
    pub fn sort_key(&self) -> (i8, u8) {
        match self {
            Card::Special(SpecialKind::Dog) => (-1, 0),
            Card::Special(SpecialKind::MahJong) => (0, 1),
            Card::Normal { rank, suit } => {
                let suit_order = match suit {
                    Suit::Jade => 0,
                    Suit::Sword => 1,
                    Suit::Pagoda => 2,
                    Suit::Star => 3,
                };
                (*rank as i8, suit_order)
            }
            Card::Special(SpecialKind::Phoenix) => (15, 0),
            Card::Special(SpecialKind::Dragon) => (16, 0),
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id())
    }
}

// ---------------------------------------------------------------------------
// Combination types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComboKind {
    Single,
    Pair,
    Triple,
    FullHouse,
    Straight,
    ConsecutivePairs,
    FourOfAKindBomb,
    StraightFlushBomb,
}

/// A validated combination of cards.
///
/// `rank` is an f32 to accommodate Phoenix-as-single (e.g. 10.5).
/// `length` is the number of cards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Combo {
    pub kind: ComboKind,
    pub cards: Vec<Card>,
    pub rank: f32,
    pub length: usize,
}

// ---------------------------------------------------------------------------
// Seats & Teams
// ---------------------------------------------------------------------------

/// Seat indices 0..3 around the table, clockwise.
/// Team 0 = seats {0, 2}, Team 1 = seats {1, 3}.
pub type Seat = usize;
pub type Team = usize;

pub fn team_for_seat(seat: Seat) -> Team {
    seat % 2
}

pub fn partner_seat(seat: Seat) -> Seat {
    (seat + 2) % 4
}

pub fn left_seat(seat: Seat) -> Seat {
    (seat + 1) % 4
}

pub fn right_seat(seat: Seat) -> Seat {
    (seat + 3) % 4
}

// ---------------------------------------------------------------------------
// Game phases
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Dealing,
    GrandTichuWindow,
    Passing,
    Playing,
    RoundEnd,
    GameEnd,
}

// ---------------------------------------------------------------------------
// Tichu declarations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TichuCall {
    Grand,
    Small,
}

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Clockwise,
    CounterClockwise,
}
