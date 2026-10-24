//! What one seat is allowed to see. Bots only ever get a `PlayerView`, so a
//! bot can't peek at other hands by accident. (It is roughly the server's
//! `ClientGameState`, plus this round's action log.)

use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::engine::{Action, GameState, TrickPlay};
use crate::types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicPlayer {
    pub card_count: usize,
    pub tichu: TichuCall,
    /// 0 = still in, otherwise 1-based finishing position.
    pub out_order: u8,
    pub has_played_first_card: bool,
    pub grand_decided: bool,
    pub has_passed: bool,
    pub trick_count: usize,
    pub captured_points: i32,
}

impl PublicPlayer {
    pub fn is_out(&self) -> bool {
        self.out_order != 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerView {
    pub seat: Seat,
    pub phase: Phase,
    pub settings: Settings,
    pub round_number: u32,
    /// Cumulative scores, indexed by team (team = seat % 2).
    pub scores: [i32; 2],
    /// My hand (8 cards during the Grand Tichu window).
    pub hand: Vec<Card>,
    pub players: [PublicPlayer; 4],
    pub current_trick: Option<Combo>,
    pub trick_plays: Vec<TrickPlay>,
    pub last_played_by: Option<Seat>,
    pub turn: Seat,
    pub pass_count: usize,
    pub wish: Option<Rank>,
    pub wish_pending: bool,
    pub trick_countdown: Option<Seat>,
    pub dragon_giveaway_by: Option<Seat>,
    /// Every card played this round, in order.
    pub played_cards: Vec<Card>,
    /// What I passed (None until I pass).
    pub my_pass: Option<PassInfo>,
    /// What I received, and from whom (empty until passes are applied).
    pub received: Vec<(Card, Seat)>,
    /// This round's actions so far; other seats' card passes are hidden.
    pub log: Vec<Action>,
}

impl GameState {
    pub fn view(&self, seat: Seat) -> PlayerView {
        let players = [0, 1, 2, 3].map(|s| {
            let p = &self.players[s];
            PublicPlayer {
                card_count: p.hand.len(),
                tichu: p.tichu,
                out_order: p.out_order,
                has_played_first_card: p.has_played_first_card,
                grand_decided: p.grand_decided,
                has_passed: p.pass.is_some(),
                trick_count: p.tricks_won.len(),
                captured_points: p.captured_points(),
            }
        });
        let log = self
            .log
            .iter()
            .filter(|a| !matches!(a, Action::PassCards { seat: s, .. } if *s != seat))
            .cloned()
            .collect();
        PlayerView {
            seat,
            phase: self.phase,
            settings: self.settings,
            round_number: self.round_number,
            scores: self.scores,
            hand: self.players[seat].hand.clone(),
            players,
            current_trick: self.current_trick.clone(),
            trick_plays: self.trick_plays.clone(),
            last_played_by: self.last_played_by,
            turn: self.turn,
            pass_count: self.pass_count,
            wish: self.wish,
            wish_pending: self.wish_pending,
            trick_countdown: self.trick_countdown,
            dragon_giveaway_by: self.dragon_giveaway_by,
            played_cards: self.played_cards.clone(),
            my_pass: self.players[seat].pass,
            received: self.received[seat].clone(),
            log,
        }
    }
}

impl PlayerView {
    pub fn me(&self) -> &PublicPlayer {
        &self.players[self.seat]
    }
    pub fn partner(&self) -> Seat {
        partner_seat(self.seat)
    }
    pub fn opponents(&self) -> [Seat; 2] {
        opponents(self.seat)
    }
    pub fn my_team(&self) -> Team {
        team_for_seat(self.seat)
    }
    pub fn is_teammate(&self, seat: Seat) -> bool {
        team_for_seat(seat) == self.my_team()
    }
    /// The seat after `seat` in play order.
    pub fn next_seat(&self, seat: Seat) -> Seat {
        if self.settings.clockwise { left_seat(seat) } else { right_seat(seat) }
    }
    pub fn is_leading(&self) -> bool {
        self.current_trick.is_none()
    }
    /// Points in the cards on the table.
    pub fn trick_points(&self) -> i32 {
        self.trick_plays.iter().map(|p| crate::scoring::total_points(&p.combo.cards)).sum()
    }
    /// Cards I can't see: not in my hand and not yet played. These are in the
    /// other three hands (or, before the second deal, still in the deck).
    pub fn unseen(&self) -> CardSet {
        CardSet::FULL.minus(CardSet::from_cards(&self.hand)).minus(CardSet::from_cards(&self.played_cards))
    }
    /// Unplayed cards I passed, with the seat that holds them.
    pub fn known_holdings(&self) -> Vec<(Card, Seat)> {
        let Some(pass) = self.my_pass else { return vec![] };
        if self.phase == Phase::Passing {
            return vec![];
        }
        let played = CardSet::from_cards(&self.played_cards);
        pass.cards()
            .into_iter()
            .filter(|c| !played.contains(*c))
            .map(|c| (c, pass.recipient(self.seat, c).unwrap()))
            .collect()
    }

    /// A random guess at all four hands consistent with what I've seen: my
    /// own hand, the other seats' card counts, and the unplayed cards I passed
    /// (which I know the location of). Uniform otherwise — a real bot will
    /// want to weight this by what the others have played and passed on.
    pub fn sample_hands<R: Rng + ?Sized>(&self, rng: &mut R) -> [Vec<Card>; 4] {
        let mut hands: [Vec<Card>; 4] = Default::default();
        hands[self.seat] = self.hand.clone();
        let mut pool = self.unseen();
        for (card, seat) in self.known_holdings() {
            if !self.players[seat].is_out() && hands[seat].len() < self.players[seat].card_count {
                hands[seat].push(card);
                pool.remove(card);
            }
        }
        let mut pool = pool.cards();
        pool.shuffle(rng);
        for seat in 0..4 {
            if seat == self.seat {
                continue;
            }
            while hands[seat].len() < self.players[seat].card_count {
                hands[seat].push(pool.pop().expect("card counts exceed unseen cards"));
            }
        }
        hands
    }
}
