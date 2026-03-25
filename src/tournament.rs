use crate::player::Player;
use crate::runner::run_game;

pub struct TournamentConfig {
    /// Number of games per seating arrangement.
    pub num_games: usize,
    /// If true, play another set with teams swapped to eliminate seat bias.
    pub swap_seats: bool,
}

#[derive(Debug, Clone)]
pub struct TournamentResult {
    pub wins_a: usize,
    pub wins_b: usize,
    pub draws: usize,
    pub total_games: usize,
    pub win_rate_a: f64,
    pub significant: bool,
}

impl std::fmt::Display for TournamentResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "games={}", self.total_games)?;
        writeln!(f, "wins_a={}", self.wins_a)?;
        writeln!(f, "wins_b={}", self.wins_b)?;
        writeln!(f, "draws={}", self.draws)?;
        writeln!(f, "win_rate_a={:.4}", self.win_rate_a)?;
        write!(f, "significant={}", self.significant)
    }
}

/// Run a tournament between two bot factories.
/// Factory A produces Team A players (seats 0, 2).
/// Factory B produces Team B players (seats 1, 3).
pub fn run_tournament(
    config: &TournamentConfig,
    factory_a: &dyn Fn() -> Box<dyn Player>,
    factory_b: &dyn Fn() -> Box<dyn Player>,
) -> TournamentResult {
    let mut wins_a = 0usize;
    let mut wins_b = 0usize;
    let mut draws = 0usize;

    // Normal seating: A = seats 0,2; B = seats 1,3.
    for _ in 0..config.num_games {
        let mut a0 = factory_a();
        let mut a1 = factory_a();
        let mut b0 = factory_b();
        let mut b1 = factory_b();
        let mut players: [&mut dyn Player; 4] = [
            a0.as_mut(),
            b0.as_mut(),
            a1.as_mut(),
            b1.as_mut(),
        ];
        let result = run_game(&mut players);
        match result.winner {
            Some(0) => wins_a += 1, // Team 0 = seats 0,2 = A
            Some(1) => wins_b += 1,
            _ => draws += 1,
        }
    }

    // Swapped seating: A = seats 1,3; B = seats 0,2.
    if config.swap_seats {
        for _ in 0..config.num_games {
            let mut a0 = factory_a();
            let mut a1 = factory_a();
            let mut b0 = factory_b();
            let mut b1 = factory_b();
            let mut players: [&mut dyn Player; 4] = [
                b0.as_mut(),
                a0.as_mut(),
                b1.as_mut(),
                a1.as_mut(),
            ];
            let result = run_game(&mut players);
            match result.winner {
                Some(0) => wins_b += 1, // Team 0 = seats 0,2 = B (swapped)
                Some(1) => wins_a += 1,
                _ => draws += 1,
            }
        }
    }

    let total = wins_a + wins_b + draws;
    let win_rate_a = if total > 0 {
        wins_a as f64 / (wins_a + wins_b).max(1) as f64
    } else {
        0.5
    };

    // Simple significance test: 95% CI for binomial proportion.
    // CI = p ± 1.96 * sqrt(p*(1-p)/n)
    let n = (wins_a + wins_b) as f64; // exclude draws for win rate calc
    let significant = if n >= 10.0 {
        let margin = 1.96 * (win_rate_a * (1.0 - win_rate_a) / n).sqrt();
        let ci_low = win_rate_a - margin;
        let ci_high = win_rate_a + margin;
        ci_low > 0.5 || ci_high < 0.5 // CI excludes 0.50
    } else {
        false
    };

    TournamentResult {
        wins_a,
        wins_b,
        draws,
        total_games: total,
        win_rate_a,
        significant,
    }
}
