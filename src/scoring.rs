use crate::types::*;

/// Point value of a single card.
pub fn card_points(card: &Card) -> i32 {
    match card {
        Card::Normal { rank: 5, .. } => 5,
        Card::Normal { rank: 10, .. } => 10,
        Card::Normal { rank: 13, .. } => 10, // Kings
        Card::Special(SpecialKind::Dragon) => 25,
        Card::Special(SpecialKind::Phoenix) => -25,
        _ => 0,
    }
}

/// Total points in a collection of cards.
pub fn total_points(cards: &[Card]) -> i32 {
    cards.iter().map(|c| card_points(c)).sum()
}

/// Score a completed round. Returns (team0_points, team1_points) including
/// card points and tichu bonuses.
pub fn score_round(
    out_order: &[Seat],           // seats in order they went out
    tricks_won: &[Vec<Vec<Card>>; 4], // tricks_won[seat] = list of tricks (each a vec of cards)
    hands: &[Vec<Card>; 4],       // remaining cards in hand (empty for players who went out)
    tichu_calls: &[Option<TichuCall>; 4],
) -> (i32, i32) {
    let mut team_points = [0i32; 2];

    let first_out = out_order[0];
    let first_team = team_for_seat(first_out);

    // Check for double victory: first two out are on the same team.
    let double_victory = out_order.len() >= 2 && team_for_seat(out_order[1]) == first_team;

    if double_victory {
        team_points[first_team] = 200;
        // Opposing team gets 0 card points.
    } else {
        // Last player out (or still holding cards).
        let last_seat = if out_order.len() == 4 {
            out_order[3]
        } else {
            // Find the seat not in out_order (or last remaining).
            (0..4).find(|s| !out_order.contains(s)).unwrap_or(out_order[3])
        };

        let last_team = team_for_seat(last_seat);
        let opposing_team = 1 - last_team;

        // Each player's trick points go to their own team.
        for seat in 0..4 {
            let trick_pts: i32 = tricks_won[seat]
                .iter()
                .flat_map(|trick| trick.iter())
                .map(|c| card_points(c))
                .sum();

            if seat == last_seat {
                // Last player's tricks go to the first-out player's team.
                team_points[first_team] += trick_pts;
            } else {
                team_points[team_for_seat(seat)] += trick_pts;
            }
        }

        // Last player's remaining hand cards go to opposing team.
        let hand_pts: i32 = hands[last_seat].iter().map(|c| card_points(c)).sum();
        team_points[opposing_team] += hand_pts;
    }

    // Tichu bonuses.
    for seat in 0..4 {
        if let Some(call) = tichu_calls[seat] {
            let succeeded = out_order.first() == Some(&seat);
            let bonus = match call {
                TichuCall::Grand => {
                    if succeeded {
                        200
                    } else {
                        -200
                    }
                }
                TichuCall::Small => {
                    if succeeded {
                        100
                    } else {
                        -100
                    }
                }
            };
            team_points[team_for_seat(seat)] += bonus;
        }
    }

    (team_points[0], team_points[1])
}
