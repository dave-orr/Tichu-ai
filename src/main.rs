use tichu_engine::engine::GameState;
use tichu_engine::types::*;

fn main() {
    let mut game = GameState::new();
    game.start_new_round();

    println!("Tichu Engine - New Round Started");
    println!("Phase: {:?}", game.phase);
    println!();

    for seat in 0..4 {
        println!(
            "Seat {} (Team {}): {} cards - {:?}",
            seat,
            team_for_seat(seat),
            game.hands[seat].len(),
            game.hands[seat]
                .iter()
                .map(|c| c.id())
                .collect::<Vec<_>>()
        );
    }

    // Auto-decide Grand Tichu (all decline) for demo.
    for seat in 0..4 {
        game.decide_grand_tichu(seat, false).unwrap();
    }
    println!("\nAll players declined Grand Tichu.");
    println!("Phase: {:?}", game.phase);

    for seat in 0..4 {
        println!(
            "Seat {} now has {} cards",
            seat,
            game.hands[seat].len()
        );
    }

    println!("\nEngine ready for passing phase.");
}
