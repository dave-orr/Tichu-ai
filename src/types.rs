use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Suits & Ranks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Suit {
    Jade,
    Sword,
    Pagoda,
    Star,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Jade, Suit::Sword, Suit::Pagoda, Suit::Star];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Suit::Jade => "jade",
            Suit::Sword => "sword",
            Suit::Pagoda => "pagoda",
            Suit::Star => "star",
        }
    }
}

/// Normal ranks 2..=14  (J=11, Q=12, K=13, A=14). The Mah Jong counts as 1
/// inside straights.
pub type Rank = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpecialKind {
    #[serde(rename = "mahjong")]
    MahJong,
    Dog,
    Phoenix,
    Dragon,
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

/// A card. Serializes to the same JSON as the TypeScript engine / server API:
/// `{"type":"normal","suit":"jade","rank":7}` or `{"type":"special","name":"dragon"}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "CardJson", try_from = "CardJson")]
pub enum Card {
    Normal { suit: Suit, rank: Rank },
    Special(SpecialKind),
}

pub const MAH_JONG: Card = Card::Special(SpecialKind::MahJong);
pub const DOG: Card = Card::Special(SpecialKind::Dog);
pub const PHOENIX: Card = Card::Special(SpecialKind::Phoenix);
pub const DRAGON: Card = Card::Special(SpecialKind::Dragon);

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum CardJson {
    Normal { suit: Suit, rank: Rank },
    Special { name: SpecialKind },
}

impl From<Card> for CardJson {
    fn from(c: Card) -> Self {
        match c {
            Card::Normal { suit, rank } => CardJson::Normal { suit, rank },
            Card::Special(name) => CardJson::Special { name },
        }
    }
}

impl TryFrom<CardJson> for Card {
    type Error = String;
    fn try_from(c: CardJson) -> Result<Self, String> {
        match c {
            CardJson::Normal { suit, rank } if (2..=14).contains(&rank) => Ok(Card::Normal { suit, rank }),
            CardJson::Normal { rank, .. } => Err(format!("invalid rank {rank}")),
            CardJson::Special { name } => Ok(Card::Special(name)),
        }
    }
}

impl Card {
    pub fn normal(suit: Suit, rank: Rank) -> Card {
        Card::Normal { suit, rank }
    }

    pub fn is_special(&self) -> bool {
        matches!(self, Card::Special(_))
    }

    pub fn is(&self, kind: SpecialKind) -> bool {
        *self == Card::Special(kind)
    }

    /// Rank of a normal card (None for specials).
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

    /// Dense index 0..56: normal cards are `suit * 13 + rank - 2`, then
    /// Mah Jong 52, Dog 53, Phoenix 54, Dragon 55.
    pub fn index(&self) -> usize {
        match self {
            Card::Normal { suit, rank } => suit.index() * 13 + (*rank as usize - 2),
            Card::Special(SpecialKind::MahJong) => 52,
            Card::Special(SpecialKind::Dog) => 53,
            Card::Special(SpecialKind::Phoenix) => 54,
            Card::Special(SpecialKind::Dragon) => 55,
        }
    }

    pub fn from_index(i: usize) -> Card {
        match i {
            0..=51 => Card::Normal { suit: Suit::ALL[i / 13], rank: (i % 13) as u8 + 2 },
            52 => MAH_JONG,
            53 => DOG,
            54 => PHOENIX,
            55 => DRAGON,
            _ => panic!("card index out of range: {i}"),
        }
    }

    /// Identifier string matching TS `cardId` (`jade-7`, `phoenix`, …).
    pub fn id(&self) -> String {
        match self {
            Card::Normal { suit, rank } => format!("{}-{}", suit.name(), rank),
            Card::Special(SpecialKind::MahJong) => "mahjong".into(),
            Card::Special(SpecialKind::Dog) => "dog".into(),
            Card::Special(SpecialKind::Phoenix) => "phoenix".into(),
            Card::Special(SpecialKind::Dragon) => "dragon".into(),
        }
    }

    /// Short human-readable label: `7j`, `Ks`, `MJ`, `Dog`, `Ph`, `Dr`.
    pub fn short(&self) -> String {
        match self {
            Card::Normal { suit, rank } => {
                let r = match rank {
                    10 => "T".to_string(),
                    11 => "J".to_string(),
                    12 => "Q".to_string(),
                    13 => "K".to_string(),
                    14 => "A".to_string(),
                    n => n.to_string(),
                };
                let s = match suit {
                    Suit::Jade => 'j',
                    Suit::Sword => 's',
                    Suit::Pagoda => 'p',
                    Suit::Star => '*',
                };
                format!("{r}{s}")
            }
            Card::Special(SpecialKind::MahJong) => "MJ".into(),
            Card::Special(SpecialKind::Dog) => "Dog".into(),
            Card::Special(SpecialKind::Phoenix) => "Ph".into(),
            Card::Special(SpecialKind::Dragon) => "Dr".into(),
        }
    }

    /// Sort value matching TS `cardSortValue` (Dog 0, Mah Jong 1, 2..14,
    /// Phoenix 15, Dragon 16); ties broken by suit so sorting is total.
    pub fn sort_key(&self) -> (u8, u8) {
        match self {
            Card::Special(SpecialKind::Dog) => (0, 0),
            Card::Special(SpecialKind::MahJong) => (1, 0),
            Card::Normal { rank, suit } => (*rank, suit.index() as u8),
            Card::Special(SpecialKind::Phoenix) => (15, 0),
            Card::Special(SpecialKind::Dragon) => (16, 0),
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short())
    }
}

pub fn cards_str(cards: &[Card]) -> String {
    cards.iter().map(|c| c.short()).collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Card sets (bitmask over Card::index)
// ---------------------------------------------------------------------------

/// A set of cards as a 56-bit mask. Cheap to copy; handy for card counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CardSet(pub u64);

impl CardSet {
    pub const EMPTY: CardSet = CardSet(0);
    pub const FULL: CardSet = CardSet((1u64 << 56) - 1);

    pub fn from_cards(cards: &[Card]) -> CardSet {
        let mut s = CardSet::EMPTY;
        for c in cards {
            s.insert(*c);
        }
        s
    }
    pub fn insert(&mut self, c: Card) {
        self.0 |= 1u64 << c.index();
    }
    pub fn remove(&mut self, c: Card) {
        self.0 &= !(1u64 << c.index());
    }
    pub fn contains(&self, c: Card) -> bool {
        self.0 & (1u64 << c.index()) != 0
    }
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
    pub fn union(self, o: CardSet) -> CardSet {
        CardSet(self.0 | o.0)
    }
    pub fn minus(self, o: CardSet) -> CardSet {
        CardSet(self.0 & !o.0)
    }
    pub fn intersect(self, o: CardSet) -> CardSet {
        CardSet(self.0 & o.0)
    }
    pub fn cards(&self) -> Vec<Card> {
        let mut v = Vec::with_capacity(self.len());
        let mut bits = self.0;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            v.push(Card::from_index(i));
            bits &= bits - 1;
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Combination types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

impl ComboKind {
    pub fn is_bomb(self) -> bool {
        matches!(self, ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb)
    }
}

/// A validated combination of cards.
///
/// `rank` is an f32 to accommodate Phoenix-as-single (e.g. 10.5).
/// `length` is the number of cards. `phoenix_as` records the rank the Phoenix
/// stands for when the reading matters (straights, 2+2+Phoenix full houses);
/// pass it back to the engine when playing the combo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Combo {
    #[serde(rename = "type")]
    pub kind: ComboKind,
    pub cards: Vec<Card>,
    pub rank: f32,
    pub length: usize,
    #[serde(rename = "phoenixAs", default, skip_serializing_if = "Option::is_none")]
    pub phoenix_as: Option<u8>,
}

impl Combo {
    pub fn is_bomb(&self) -> bool {
        self.kind.is_bomb()
    }
    pub fn contains(&self, kind: SpecialKind) -> bool {
        self.cards.iter().any(|c| c.is(kind))
    }
    pub fn contains_rank(&self, rank: Rank) -> bool {
        self.cards.iter().any(|c| c.rank() == Some(rank))
    }
    pub fn card_set(&self) -> CardSet {
        CardSet::from_cards(&self.cards)
    }
}

impl fmt::Display for Combo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({}) [{}]", self.kind, self.rank, cards_str(&self.cards))
    }
}

// ---------------------------------------------------------------------------
// Seats & Teams
// ---------------------------------------------------------------------------

/// Seat indices 0..3. Team 0 = seats {0, 2}, Team 1 = seats {1, 3}.
/// Naming matches the TS engine: "right" is seat+1, "left" is seat+3. Default
/// play (counterclockwise) moves to the right, i.e. 0 → 1 → 2 → 3.
pub type Seat = usize;
pub type Team = usize;

pub fn team_for_seat(seat: Seat) -> Team {
    seat % 2
}

pub fn partner_seat(seat: Seat) -> Seat {
    (seat + 2) % 4
}

pub fn left_seat(seat: Seat) -> Seat {
    (seat + 3) % 4
}

pub fn right_seat(seat: Seat) -> Seat {
    (seat + 1) % 4
}

pub fn opponents(seat: Seat) -> [Seat; 2] {
    [right_seat(seat), left_seat(seat)]
}

// ---------------------------------------------------------------------------
// Settings, phases, calls
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// Play direction: false (default) = counterclockwise, turn passes to seat+1.
    pub clockwise: bool,
    pub target_score: i32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { clockwise: false, target_score: 1000 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Phase {
    GrandTichuWindow,
    Passing,
    Playing,
    RoundEnd,
    GameEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TichuCall {
    None,
    Small,
    Grand,
}

/// Cards a player passes: one to each other seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassInfo {
    /// To seat+3.
    pub left: Card,
    /// To seat+2.
    pub partner: Card,
    /// To seat+1.
    pub right: Card,
}

impl PassInfo {
    pub fn cards(&self) -> [Card; 3] {
        [self.left, self.partner, self.right]
    }
    /// Which seat receives `card` from `from`, if it is one of the three.
    pub fn recipient(&self, from: Seat, card: Card) -> Option<Seat> {
        if card == self.left {
            Some(left_seat(from))
        } else if card == self.partner {
            Some(partner_seat(from))
        } else if card == self.right {
            Some(right_seat(from))
        } else {
            None
        }
    }
}
