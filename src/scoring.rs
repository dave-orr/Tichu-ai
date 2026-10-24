use serde::{Deserialize, Serialize};

use crate::types::*;

/// Point value of a single card.
pub fn card_points(card: &Card) -> i32 {
    match card {
        Card::Normal { rank: 5, .. } => 5,
        Card::Normal { rank: 10, .. } | Card::Normal { rank: 13, .. } => 10,
        Card::Special(SpecialKind::Dragon) => 25,
        Card::Special(SpecialKind::Phoenix) => -25,
        _ => 0,
    }
}

/// Total points in a collection of cards.
pub fn total_points(cards: &[Card]) -> i32 {
    cards.iter().map(card_points).sum()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundResult {
    /// Card points (or 200/0 for a 1-2 finish), per team.
    pub card_points: [i32; 2],
    pub tichu_bonuses: [i32; 2],
    pub double_victory: Option<Team>,
    /// Seats in finishing order; seats that never went out come last, in seat order.
    pub out_order: [Seat; 4],
    pub tichu_calls: [TichuCall; 4],
}

impl RoundResult {
    pub fn total(&self, team: Team) -> i32 {
        self.card_points[team] + self.tichu_bonuses[team]
    }
    /// Team 0's total minus team 1's.
    pub fn diff(&self) -> i32 {
        self.total(0) - self.total(1)
    }
}

/// Score a finished round (mirrors TS `scoreRound`). `out_order[seat]` is
/// 1-based finishing position, 0 for a seat that never went out.
pub fn score_round(
    out_order: &[u8; 4],
    tricks_won: &[Vec<Vec<Card>>; 4],
    hands: &[Vec<Card>; 4],
    tichu_calls: &[TichuCall; 4],
) -> RoundResult {
    let mut order: Vec<Seat> = (0..4).collect();
    order.sort_by_key(|&s| if out_order[s] == 0 { u8::MAX } else { out_order[s] });
    let order: [Seat; 4] = [order[0], order[1], order[2], order[3]];
    let (first, second, last) = (order[0], order[1], order[3]);

    let mut card_points = [0i32; 2];
    let double_victory = if team_for_seat(first) == team_for_seat(second) {
        card_points[team_for_seat(first)] = 200;
        Some(team_for_seat(first))
    } else {
        for seat in 0..4 {
            let pts: i32 = tricks_won[seat].iter().map(|t| total_points(t)).sum();
            // The last player's tricks go to whoever went out first.
            let team = if seat == last { team_for_seat(first) } else { team_for_seat(seat) };
            card_points[team] += pts;
            // Cards left in hand go to the last player's opponents.
            card_points[1 - team_for_seat(last)] += total_points(&hands[seat]);
        }
        None
    };

    let mut tichu_bonuses = [0i32; 2];
    for seat in 0..4 {
        let made = out_order[seat] == 1;
        let bonus = match tichu_calls[seat] {
            TichuCall::Grand => 200,
            TichuCall::Small => 100,
            TichuCall::None => 0,
        };
        tichu_bonuses[team_for_seat(seat)] += if made { bonus } else { -bonus };
    }

    RoundResult { card_points, tichu_bonuses, double_victory, out_order: order, tichu_calls: *tichu_calls }
}

/// A team has reached the target and the scores aren't tied.
pub fn is_game_over(scores: [i32; 2], target: i32) -> bool {
    (scores[0] >= target || scores[1] >= target) && scores[0] != scores[1]
}
