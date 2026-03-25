use crate::player::Player;
use crate::runner::{run_game, GameResult};
use crate::types::TichuCall;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub struct HarnessConfig {
    pub num_games: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AggregateStats {
    pub games_played: usize,
    pub team_wins: [usize; 2],
    pub draws: usize,
    pub avg_final_scores: [f64; 2],
    pub avg_rounds_per_game: f64,
    pub tichu_attempts: [usize; 4],
    pub tichu_successes: [usize; 4],
    pub grand_tichu_attempts: [usize; 4],
    pub grand_tichu_successes: [usize; 4],
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

pub struct HarnessResult {
    pub stats: AggregateStats,
    pub game_results: Vec<GameResult>,
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Run `config.num_games` games using players created by `player_factory`.
/// Returns aggregate statistics and full game histories.
pub fn run_harness(
    config: HarnessConfig,
    player_factory: &dyn Fn() -> [Box<dyn Player>; 4],
) -> HarnessResult {
    let mut game_results = Vec::with_capacity(config.num_games);
    let mut team_wins = [0usize; 2];
    let mut draws = 0usize;
    let mut total_scores = [0i64; 2];
    let mut total_rounds = 0usize;
    let mut tichu_attempts = [0usize; 4];
    let mut tichu_successes = [0usize; 4];
    let mut grand_tichu_attempts = [0usize; 4];
    let mut grand_tichu_successes = [0usize; 4];

    for _ in 0..config.num_games {
        let mut boxed_players = player_factory();
        let [ref mut p0, ref mut p1, ref mut p2, ref mut p3] = boxed_players;
        let mut players: [&mut dyn Player; 4] = [
            p0.as_mut(),
            p1.as_mut(),
            p2.as_mut(),
            p3.as_mut(),
        ];
        let result = run_game(&mut players);

        // Accumulate stats.
        match result.winner {
            Some(0) => team_wins[0] += 1,
            Some(1) => team_wins[1] += 1,
            _ => draws += 1,
        }
        total_scores[0] += result.final_scores[0] as i64;
        total_scores[1] += result.final_scores[1] as i64;
        total_rounds += result.rounds.len();

        // Tichu stats from each round.
        for round in &result.rounds {
            for seat in 0..4 {
                match round.tichu_calls[seat] {
                    Some(TichuCall::Grand) => {
                        grand_tichu_attempts[seat] += 1;
                        if !round.out_order.is_empty() && round.out_order[0] == seat {
                            grand_tichu_successes[seat] += 1;
                        }
                    }
                    Some(TichuCall::Small) => {
                        tichu_attempts[seat] += 1;
                        if !round.out_order.is_empty() && round.out_order[0] == seat {
                            tichu_successes[seat] += 1;
                        }
                    }
                    None => {}
                }
            }
        }

        game_results.push(result);
    }

    let n = config.num_games as f64;
    let stats = AggregateStats {
        games_played: config.num_games,
        team_wins,
        draws,
        avg_final_scores: [
            total_scores[0] as f64 / n,
            total_scores[1] as f64 / n,
        ],
        avg_rounds_per_game: total_rounds as f64 / n,
        tichu_attempts,
        tichu_successes,
        grand_tichu_attempts,
        grand_tichu_successes,
    };

    HarnessResult {
        stats,
        game_results,
    }
}
