use crate::engine::{Action, GameState};
use crate::player::Player;
use crate::types::*;

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ActionRecord {
    pub seat: Seat,
    pub action: Action,
    pub mah_jong_wish: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct RoundRecord {
    pub initial_hands: [Vec<Card>; 4],
    pub tichu_calls: [Option<TichuCall>; 4],
    pub passes: [[Card; 3]; 4],
    pub actions: Vec<ActionRecord>,
    pub out_order: Vec<Seat>,
    pub round_scores: (i32, i32),
    pub cumulative_scores: (i32, i32),
}

#[derive(Debug, Clone)]
pub struct GameResult {
    pub rounds: Vec<RoundRecord>,
    pub final_scores: [i32; 2],
    pub winner: Option<Team>,
}

// ---------------------------------------------------------------------------
// Game runner
// ---------------------------------------------------------------------------

/// Run a complete game (multiple rounds until target score) and return the result.
pub fn run_game(players: &mut [&mut dyn Player; 4]) -> GameResult {
    let mut game = GameState::new();
    let mut rounds = Vec::new();

    loop {
        game.start_new_round();

        for seat in 0..4 {
            players[seat].on_round_start(&game, seat);
        }

        // --- Grand Tichu decisions (players see first 8 cards) ---
        for seat in 0..4 {
            let call = players[seat].decide_grand_tichu(&game, seat);
            game.decide_grand_tichu(seat, call)
                .expect("decide_grand_tichu failed");
        }

        // After all GT decisions, remaining 6 cards are dealt.
        // Record initial hands (now all 14 cards).
        let initial_hands = game.hands.clone();

        // --- Small Tichu decisions ---
        for seat in 0..4 {
            if players[seat].decide_small_tichu(&game, seat) {
                // Ignore error if they already called Grand Tichu.
                let _ = game.call_small_tichu(seat);
            }
        }

        // --- Passing ---
        let mut pass_records = [[Card::Special(SpecialKind::MahJong); 3]; 4]; // placeholder
        for seat in 0..4 {
            let cards = players[seat].choose_pass_cards(&game, seat);
            pass_records[seat] = cards;
            game.pass_cards(seat, cards)
                .expect("pass_cards failed");
        }

        // --- Playing ---
        let mut action_records = Vec::new();

        while game.phase == Phase::Playing {
            if game.dragon_giveaway {
                let seat = game.dragon_giveaway_seat.unwrap();
                let legal = game.legal_actions();
                if legal.is_empty() {
                    break;
                }
                let (action, _) = players[seat].choose_action(&game, seat, &legal);
                action_records.push(ActionRecord {
                    seat,
                    action: action.clone(),
                    mah_jong_wish: None,
                });
                if let Action::GiveDragonTrick(target) = action {
                    game.give_dragon_trick(seat, target)
                        .expect("give_dragon_trick failed");
                } else {
                    panic!("Player returned non-GiveDragonTrick action during dragon giveaway");
                }
            } else {
                let seat = game.current_player;
                let legal = game.legal_actions();
                if legal.is_empty() {
                    break;
                }
                let (action, wish) = players[seat].choose_action(&game, seat, &legal);
                action_records.push(ActionRecord {
                    seat,
                    action: action.clone(),
                    mah_jong_wish: wish,
                });
                match &action {
                    Action::Pass => {
                        game.pass_turn(seat).expect("pass_turn failed");
                    }
                    Action::Play(combo) => {
                        game.play_cards(seat, &combo.cards, None, wish)
                            .expect("play_cards failed");
                    }
                    Action::GiveDragonTrick(_) => {
                        panic!("Player returned GiveDragonTrick outside of dragon giveaway");
                    }
                }
            }
        }

        // --- Record round ---
        let round_scores = game.round_history.last().copied().unwrap_or((0, 0));
        rounds.push(RoundRecord {
            initial_hands,
            tichu_calls: game.tichu_calls,
            passes: pass_records,
            actions: action_records,
            out_order: game.out_order.clone(),
            round_scores,
            cumulative_scores: (game.scores[0], game.scores[1]),
        });

        if game.phase == Phase::GameEnd {
            break;
        }
    }

    for seat in 0..4 {
        players[seat].on_game_end(&game, seat);
    }

    GameResult {
        rounds,
        final_scores: game.scores,
        winner: game.winner(),
    }
}
