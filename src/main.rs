use std::path::Path;

use clap::{Parser, Subcommand};

use tichu_engine::bots::heuristic::constants::HeuristicWeights;
use tichu_engine::bots::heuristic::HeuristicBot;
use tichu_engine::bots::ismcts::config::ISMCTSConfig;
use tichu_engine::bots::ismcts::ISMCTSBot;
use tichu_engine::bots::random::RandomBot;
use tichu_engine::engine::Action;
use tichu_engine::player::Player;
use tichu_engine::runner::run_game;
use tichu_engine::tournament::{run_tournament, TournamentConfig};
use tichu_engine::types::*;

#[derive(Parser)]
#[command(name = "tichu", about = "Tichu bot training harness")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single hand with play-by-play output.
    Demo,
    /// Benchmark heuristic bot vs random bots.
    Benchmark {
        #[arg(long, default_value_t = 200)]
        games: usize,
    },
    /// Run a tournament between heuristic and random bots.
    Tournament {
        #[arg(long, default_value_t = 200)]
        games: usize,
        /// Play a second set with teams swapped to eliminate seat bias.
        #[arg(long)]
        swap: bool,
    },
    /// Run ISMCTS bot vs heuristic bot.
    Ismcts {
        #[arg(long, default_value_t = 50)]
        games: usize,
        /// MCTS iterations per decision.
        #[arg(long, default_value_t = 200)]
        iterations: u32,
        /// Play a second set with teams swapped.
        #[arg(long)]
        swap: bool,
    },
    /// Run current ISMCTS bot vs baseline (for autoresearch).
    /// Uses low iterations (50) for speed. Creates baseline files on first run.
    Autoresearch {
        #[arg(long, default_value_t = 100)]
        games: usize,
        /// MCTS iterations per decision (low for speed).
        #[arg(long, default_value_t = 50)]
        iterations: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Demo => run_demo(),
        Commands::Benchmark { games } => run_benchmark(games),
        Commands::Tournament { games, swap } => run_tournament_cmd(games, swap),
        Commands::Ismcts { games, iterations, swap } => run_ismcts_vs_heuristic(games, iterations, swap),
        Commands::Autoresearch { games, iterations } => run_autoresearch(games, iterations),
    }
}

fn run_demo() {
    let mut p0 = HeuristicBot::new();
    let mut p1 = HeuristicBot::new();
    let mut p2 = HeuristicBot::new();
    let mut p3 = HeuristicBot::new();
    let mut players: [&mut dyn Player; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];

    let result = run_game(&mut players);
    let round = &result.rounds[0];

    println!("=== Single Hand: 4 Heuristic Bots ===\n");

    for seat in 0..4 {
        let team = team_for_seat(seat);
        let hand: Vec<String> = round.initial_hands[seat].iter().map(|c| c.id()).collect();
        println!("Seat {} (Team {}): {}", seat, team, hand.join(", "));
    }

    println!();
    for seat in 0..4 {
        if let Some(call) = round.tichu_calls[seat] {
            println!("Seat {} called {:?}!", seat, call);
        }
    }

    println!("\n--- Passing ---");
    for seat in 0..4 {
        let cards: Vec<String> = round.passes[seat].iter().map(|c| c.id()).collect();
        println!(
            "Seat {} passes: {} (left), {} (partner), {} (right)",
            seat, cards[0], cards[1], cards[2]
        );
    }

    println!("\n--- Play ---");
    let mut trick_num = 1;
    let mut in_trick = false;
    for (i, rec) in round.actions.iter().enumerate() {
        if !in_trick {
            println!("\nTrick {}:", trick_num);
            in_trick = true;
        }

        match &rec.action {
            Action::Pass => println!("  Seat {} passes", rec.seat),
            Action::Play(combo) => {
                let cards: Vec<String> = combo.cards.iter().map(|c| c.id()).collect();
                let wish_str = match rec.mah_jong_wish {
                    Some(r) => format!(" (wishes for {})", r),
                    None => String::new(),
                };
                println!(
                    "  Seat {} plays {:?}: [{}]{}",
                    rec.seat,
                    combo.kind,
                    cards.join(", "),
                    wish_str,
                );
            }
            Action::GiveDragonTrick(target) => {
                println!("  Seat {} gives dragon trick to Seat {}", rec.seat, target);
            }
        }

        let next_is_new_trick = if i + 1 < round.actions.len() {
            matches!(&rec.action, Action::GiveDragonTrick(_))
        } else {
            false
        };

        if next_is_new_trick {
            trick_num += 1;
            in_trick = false;
        }
    }

    println!("\n--- Results ---");
    println!(
        "Out order: {}",
        round.out_order.iter().map(|s| format!("Seat {}", s)).collect::<Vec<_>>().join(" -> ")
    );
    println!("Round scores: Team 0 = {}, Team 1 = {}", round.round_scores.0, round.round_scores.1);

    println!("\n--- Full Game ---");
    println!("Total rounds: {}", result.rounds.len());
    println!("Final scores: Team 0 = {}, Team 1 = {}", result.final_scores[0], result.final_scores[1]);
    match result.winner {
        Some(t) => println!("Winner: Team {}", t),
        None => println!("Draw!"),
    }
}

fn run_benchmark(games: usize) {
    let config = TournamentConfig {
        num_games: games,
        swap_seats: true,
    };

    let result = run_tournament(
        &config,
        &|| Box::new(HeuristicBot::new()) as Box<dyn Player>,
        &|| Box::new(RandomBot) as Box<dyn Player>,
    );

    println!("=== Benchmark: Heuristic vs Random ===");
    println!("{}", result);
}

fn run_tournament_cmd(games: usize, swap: bool) {
    let config = TournamentConfig {
        num_games: games,
        swap_seats: swap,
    };

    let result = run_tournament(
        &config,
        &|| Box::new(HeuristicBot::new()) as Box<dyn Player>,
        &|| Box::new(RandomBot) as Box<dyn Player>,
    );

    println!("{}", result);
}

fn run_ismcts_vs_heuristic(games: usize, iterations: u32, swap: bool) {
    let config = TournamentConfig {
        num_games: games,
        swap_seats: swap,
    };

    println!("=== ISMCTS ({} iterations) vs Heuristic ===", iterations);
    println!("Running {} games{}...", games, if swap { " (with swap)" } else { "" });

    let result = run_tournament(
        &config,
        &move || Box::new(ISMCTSBot::new(iterations)) as Box<dyn Player>,
        &|| Box::new(HeuristicBot::new()) as Box<dyn Player>,
    );

    println!("{}", result);
}

fn run_autoresearch(games: usize, eval_iterations: u32) {
    let weights_path = Path::new("baseline_weights.json");
    let config_path = Path::new("baseline_config.json");

    // If baseline files don't exist, create them from current defaults and exit.
    if !weights_path.exists() || !config_path.exists() {
        let weights = HeuristicWeights::default();
        let config = ISMCTSConfig::default();
        weights.save(weights_path).expect("Failed to save baseline weights");
        config.save(config_path).expect("Failed to save baseline config");
        println!("Created baseline files from current defaults.");
        println!("Run again after making changes to evaluate against this baseline.");
        return;
    }

    // Load baseline.
    let baseline_weights = HeuristicWeights::load(weights_path)
        .expect("Failed to load baseline_weights.json");
    let baseline_config = ISMCTSConfig::load(config_path)
        .expect("Failed to load baseline_config.json");

    // Override both to use eval_iterations for speed.
    let mut challenger_config = ISMCTSConfig::default();
    challenger_config.num_iterations = eval_iterations;

    let mut baseline_eval_config = baseline_config.clone();
    baseline_eval_config.num_iterations = eval_iterations;

    let tournament_config = TournamentConfig {
        num_games: games,
        swap_seats: true,
    };

    eprintln!(
        "=== Autoresearch: Current vs Baseline ({} iterations, {} games with swap) ===",
        eval_iterations, games
    );

    let result = run_tournament(
        &tournament_config,
        &move || {
            Box::new(ISMCTSBot::from_config(
                challenger_config.clone(),
                HeuristicWeights::default(),
            )) as Box<dyn Player>
        },
        &move || {
            Box::new(ISMCTSBot::from_config(
                baseline_eval_config.clone(),
                baseline_weights.clone(),
            )) as Box<dyn Player>
        },
    );

    // Print parseable output to stdout (eprintln for human-readable).
    println!("{}", result);
}
