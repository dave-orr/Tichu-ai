use crate::combinations::*;
use crate::deck::*;
use crate::scoring::*;
use crate::types::*;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameError {
    WrongPhase,
    NotYourTurn,
    InvalidCards,
    InvalidCombo,
    CannotBeatCurrentTrick,
    MustFulfillWish,
    CannotPass,
    InvalidSeat,
    InvalidDragonTarget,
    AlreadyCalledTichu,
    MustLeadWithMahJong,
    DogMustBeLead,
    CardNotInHand,
}

impl std::fmt::Display for GameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for GameError {}

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub phase: Phase,
    pub hands: [Vec<Card>; 4],
    pub tricks_won: [Vec<Vec<Card>>; 4], // tricks_won[seat] = list of captured card piles
    pub current_trick: Option<Combo>,    // the combo to beat (None = lead)
    pub current_trick_cards: Vec<Card>,  // all cards in the current trick pile
    pub current_player: Seat,
    pub pass_count: usize,               // consecutive passes
    pub last_played_by: Option<Seat>,
    pub out_order: Vec<Seat>,            // seats in the order they went out
    pub tichu_calls: [Option<TichuCall>; 4],
    pub grand_tichu_decided: [bool; 4],
    pub passes: [Option<[Card; 3]>; 4],  // cards each player passes (before applying)
    pub received_cards: [Vec<Card>; 4],
    pub mah_jong_wish: Option<u8>,        // wished rank (None = no active wish)
    pub dragon_giveaway: bool,            // waiting for dragon trick recipient
    pub dragon_giveaway_seat: Option<Seat>,
    pub scores: [i32; 2],                // cumulative team scores
    pub round_history: Vec<(i32, i32)>,  // per-round scores
    pub target_score: i32,
    pub direction: Direction,
    pub first_card_played: bool,         // has any card been played this round?
    #[serde(skip)]
    _remaining_cards: Option<Vec<Card>>, // stashed between deal phases
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            phase: Phase::Dealing,
            hands: [vec![], vec![], vec![], vec![]],
            tricks_won: [vec![], vec![], vec![], vec![]],
            current_trick: None,
            current_trick_cards: Vec::new(),
            current_player: 0,
            pass_count: 0,
            last_played_by: None,
            out_order: Vec::new(),
            tichu_calls: [None; 4],
            grand_tichu_decided: [false; 4],
            passes: [None, None, None, None],
            received_cards: [vec![], vec![], vec![], vec![]],
            mah_jong_wish: None,
            dragon_giveaway: false,
            dragon_giveaway_seat: None,
            scores: [0, 0],
            round_history: Vec::new(),
            target_score: 1000,
            direction: Direction::Clockwise,
            first_card_played: false,
            _remaining_cards: None,
        }
    }

    // -----------------------------------------------------------------------
    // Phase: Dealing
    // -----------------------------------------------------------------------

    /// Start a new round: shuffle deck, deal first 8 cards to each player.
    pub fn start_new_round(&mut self) {
        let mut deck = create_deck();
        shuffle(&mut deck);

        self.hands = [vec![], vec![], vec![], vec![]];
        self.tricks_won = [vec![], vec![], vec![], vec![]];
        self.current_trick = None;
        self.current_trick_cards = Vec::new();
        self.pass_count = 0;
        self.last_played_by = None;
        self.out_order = Vec::new();
        self.tichu_calls = [None; 4];
        self.grand_tichu_decided = [false; 4];
        self.passes = [None, None, None, None];
        self.received_cards = [vec![], vec![], vec![], vec![]];
        self.mah_jong_wish = None;
        self.dragon_giveaway = false;
        self.dragon_giveaway_seat = None;
        self.first_card_played = false;

        // Deal first 8 cards.
        for seat in 0..4 {
            self.hands[seat] = deck[seat * 8..(seat + 1) * 8].to_vec();
            sort_hand(&mut self.hands[seat]);
        }

        // Store remaining 24 cards temporarily (will be dealt after Grand Tichu window).
        // We stash them in a deterministic way.
        self._remaining_cards = Some(deck[32..].to_vec());
        self.phase = Phase::GrandTichuWindow;
    }

    /// Deal remaining 6 cards to each player (after Grand Tichu decisions).
    fn deal_remaining(&mut self, remaining: Vec<Card>) {
        for seat in 0..4 {
            let start = seat * 6;
            let end = start + 6;
            self.hands[seat].extend_from_slice(&remaining[start..end]);
            sort_hand(&mut self.hands[seat]);
        }
        self.phase = Phase::Passing;
    }

    // -----------------------------------------------------------------------
    // Grand Tichu
    // -----------------------------------------------------------------------

    /// Player decides on Grand Tichu (call or decline).
    pub fn decide_grand_tichu(&mut self, seat: Seat, call: bool) -> Result<(), GameError> {
        if self.phase != Phase::GrandTichuWindow {
            return Err(GameError::WrongPhase);
        }
        if seat >= 4 {
            return Err(GameError::InvalidSeat);
        }

        if call {
            self.tichu_calls[seat] = Some(TichuCall::Grand);
        }
        self.grand_tichu_decided[seat] = true;

        // If all 4 have decided, deal remaining cards.
        if self.grand_tichu_decided.iter().all(|&d| d) {
            if let Some(remaining) = self._remaining_cards.take() {
                self.deal_remaining(remaining);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Small Tichu
    // -----------------------------------------------------------------------

    /// Call Small Tichu (only allowed before the player's first card is played).
    pub fn call_small_tichu(&mut self, seat: Seat) -> Result<(), GameError> {
        if self.phase != Phase::Playing && self.phase != Phase::Passing {
            return Err(GameError::WrongPhase);
        }
        if seat >= 4 {
            return Err(GameError::InvalidSeat);
        }
        if self.tichu_calls[seat].is_some() {
            return Err(GameError::AlreadyCalledTichu);
        }
        // Small tichu: must be before the player has played any card.
        // In the TS engine: allowed before firstCardPlayed for any player,
        // but actually it's per-player. The TS checks !firstCardPlayed.
        // We'll follow the same logic: allowed before any card is played this round.
        if self.first_card_played {
            return Err(GameError::WrongPhase);
        }
        self.tichu_calls[seat] = Some(TichuCall::Small);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Passing
    // -----------------------------------------------------------------------

    /// Player submits 3 cards to pass (one to each other player).
    /// cards[0] -> left, cards[1] -> partner, cards[2] -> right
    pub fn pass_cards(&mut self, seat: Seat, cards: [Card; 3]) -> Result<(), GameError> {
        if self.phase != Phase::Passing {
            return Err(GameError::WrongPhase);
        }
        if seat >= 4 {
            return Err(GameError::InvalidSeat);
        }
        // Validate cards are in hand.
        for c in &cards {
            if !self.hands[seat].contains(c) {
                return Err(GameError::CardNotInHand);
            }
        }
        // Validate no duplicates.
        if cards[0] == cards[1] || cards[0] == cards[2] || cards[1] == cards[2] {
            return Err(GameError::InvalidCards);
        }

        self.passes[seat] = Some(cards);

        // If all 4 have submitted passes, apply them.
        if self.passes.iter().all(|p| p.is_some()) {
            self.apply_passes();
        }
        Ok(())
    }

    fn apply_passes(&mut self) {
        // Remove passed cards from each hand and deliver to recipients.
        // passes[seat] = [to_left, to_partner, to_right]
        let passes: Vec<[Card; 3]> = self.passes.iter().map(|p| p.unwrap()).collect();

        for seat in 0..4 {
            let [to_left, to_partner, to_right] = passes[seat];
            // Remove from hand.
            for card in &[to_left, to_partner, to_right] {
                if let Some(pos) = self.hands[seat].iter().position(|c| c == card) {
                    self.hands[seat].remove(pos);
                }
            }
        }

        // Deliver cards.
        for seat in 0..4 {
            let [to_left, to_partner, to_right] = passes[seat];
            let left = left_seat(seat);
            let partner = partner_seat(seat);
            let right = right_seat(seat);

            self.hands[left].push(to_left);
            self.received_cards[left].push(to_left);

            self.hands[partner].push(to_partner);
            self.received_cards[partner].push(to_partner);

            self.hands[right].push(to_right);
            self.received_cards[right].push(to_right);
        }

        // Sort all hands.
        for seat in 0..4 {
            sort_hand(&mut self.hands[seat]);
        }

        // Whoever holds MahJong starts.
        for seat in 0..4 {
            if self.hands[seat]
                .iter()
                .any(|c| matches!(c, Card::Special(SpecialKind::MahJong)))
            {
                self.current_player = seat;
                break;
            }
        }

        self.phase = Phase::Playing;
    }

    // -----------------------------------------------------------------------
    // Playing cards
    // -----------------------------------------------------------------------

    /// Play a set of cards as the current player.
    pub fn play_cards(
        &mut self,
        seat: Seat,
        cards: &[Card],
        phoenix_as: Option<u8>,
        mah_jong_wish: Option<u8>,
    ) -> Result<(), GameError> {
        if self.phase != Phase::Playing {
            return Err(GameError::WrongPhase);
        }
        if seat != self.current_player {
            return Err(GameError::NotYourTurn);
        }
        if self.dragon_giveaway {
            return Err(GameError::WrongPhase);
        }
        if self.is_out(seat) {
            return Err(GameError::NotYourTurn);
        }

        // Validate all cards are in hand.
        let mut hand_copy = self.hands[seat].clone();
        for c in cards {
            if let Some(pos) = hand_copy.iter().position(|h| h == c) {
                hand_copy.remove(pos);
            } else {
                return Err(GameError::CardNotInHand);
            }
        }

        // Dog: special handling.
        if cards.len() == 1 && matches!(cards[0], Card::Special(SpecialKind::Dog)) {
            return self.play_dog(seat);
        }

        // Identify the combination.
        let mut combo = identify_combo(cards, phoenix_as).ok_or(GameError::InvalidCombo)?;

        // If playing phoenix as single, adjust rank based on current trick.
        if cards.len() == 1 && matches!(cards[0], Card::Special(SpecialKind::Phoenix)) {
            let last_rank = self.current_trick.as_ref().map(|ct| ct.rank);
            combo.rank = single_card_rank(&cards[0], last_rank);
        }

        // Must beat current trick (if any).
        if let Some(ref current) = self.current_trick {
            if !can_beat(current, &combo) {
                return Err(GameError::CannotBeatCurrentTrick);
            }
        }

        // MahJong wish enforcement: if there's an active wish and you're not playing
        // a card of the wished rank, check if you COULD play a legal combo containing
        // the wished rank. If so, you MUST include it (or play a bomb).
        if let Some(wish_rank) = self.mah_jong_wish {
            let plays_wish = cards.iter().any(|c| match c {
                Card::Normal { rank, .. } => *rank == wish_rank,
                _ => false,
            });
            if !plays_wish {
                // Check if hand has a card of the wished rank and can form a legal combo.
                let has_wished = self.hands[seat].iter().any(|c| match c {
                    Card::Normal { rank, .. } => *rank == wish_rank,
                    _ => false,
                });
                if has_wished && can_fulfill_wish(&self.hands[seat], wish_rank, self.current_trick.as_ref()) {
                    // Allow bombs even when not fulfilling wish.
                    if !is_bomb_combo(&combo) {
                        return Err(GameError::MustFulfillWish);
                    }
                }
            }
        }

        // Remove cards from hand.
        for c in cards {
            if let Some(pos) = self.hands[seat].iter().position(|h| h == c) {
                self.hands[seat].remove(pos);
            }
        }

        // Check if MahJong was played -> prompt for wish.
        let played_mahjong = cards
            .iter()
            .any(|c| matches!(c, Card::Special(SpecialKind::MahJong)));
        if played_mahjong {
            if let Some(wish) = mah_jong_wish {
                if wish >= 2 && wish <= 14 {
                    self.mah_jong_wish = Some(wish);
                }
            }
            // If no wish provided, that's fine (player chose not to wish).
        }

        // Update trick state.
        self.current_trick_cards.extend_from_slice(cards);
        self.current_trick = Some(combo);
        self.last_played_by = Some(seat);
        self.pass_count = 0;
        self.first_card_played = true;

        // Check if player went out.
        if self.hands[seat].is_empty() {
            self.out_order.push(seat);

            // Check if wish is fulfilled (no one left with wished rank).
            self.check_wish_fulfillment();
        }

        // Check for round end.
        if self.should_round_end() {
            self.end_round();
            return Ok(());
        }

        // Advance to next player.
        self.current_player = self.get_next_active_seat(seat);

        // If the new current player is the one who last played (everyone else passed or is out),
        // award the trick.
        if Some(self.current_player) == self.last_played_by {
            self.award_trick();
        }

        Ok(())
    }

    fn play_dog(&mut self, seat: Seat) -> Result<(), GameError> {
        // Dog must be played as lead (no current trick).
        if self.current_trick.is_some() {
            return Err(GameError::DogMustBeLead);
        }

        // Remove dog from hand.
        let pos = self.hands[seat]
            .iter()
            .position(|c| matches!(c, Card::Special(SpecialKind::Dog)))
            .ok_or(GameError::CardNotInHand)?;
        self.hands[seat].remove(pos);
        self.first_card_played = true;

        // Check if player went out.
        if self.hands[seat].is_empty() {
            self.out_order.push(seat);
        }

        // Turn passes to partner (or next active if partner is out).
        let partner = partner_seat(seat);
        if self.is_out(partner) {
            self.current_player = self.get_next_active_seat(seat);
        } else {
            self.current_player = partner;
        }

        // No trick to beat - next player leads.
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Passing turn
    // -----------------------------------------------------------------------

    pub fn pass_turn(&mut self, seat: Seat) -> Result<(), GameError> {
        if self.phase != Phase::Playing {
            return Err(GameError::WrongPhase);
        }
        if seat != self.current_player {
            return Err(GameError::NotYourTurn);
        }
        if self.dragon_giveaway {
            return Err(GameError::WrongPhase);
        }

        // Cannot pass when leading.
        if self.current_trick.is_none() {
            return Err(GameError::CannotPass);
        }

        // Cannot pass if you can fulfill the wish.
        if let Some(wish_rank) = self.mah_jong_wish {
            if can_fulfill_wish(&self.hands[seat], wish_rank, self.current_trick.as_ref()) {
                return Err(GameError::MustFulfillWish);
            }
        }

        self.pass_count += 1;

        // Check if all active players (except trick winner) have passed.
        let active_count = (0..4).filter(|&s| !self.is_out(s)).count();
        // Trick is won when pass_count == active_count - 1
        // (everyone except the last player who played has passed).
        if self.pass_count >= active_count - 1 {
            self.award_trick();
            return Ok(());
        }

        self.current_player = self.get_next_active_seat(seat);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bombs (can be played out of turn)
    // -----------------------------------------------------------------------

    /// Play a bomb (can be played out of turn by any active player).
    pub fn play_bomb(
        &mut self,
        seat: Seat,
        cards: &[Card],
    ) -> Result<(), GameError> {
        if self.phase != Phase::Playing {
            return Err(GameError::WrongPhase);
        }
        if self.is_out(seat) {
            return Err(GameError::NotYourTurn);
        }

        // Validate cards in hand.
        let mut hand_copy = self.hands[seat].clone();
        for c in cards {
            if let Some(pos) = hand_copy.iter().position(|h| h == c) {
                hand_copy.remove(pos);
            } else {
                return Err(GameError::CardNotInHand);
            }
        }

        let combo = identify_combo(cards, None).ok_or(GameError::InvalidCombo)?;
        if !is_bomb_combo(&combo) {
            return Err(GameError::InvalidCombo);
        }

        // Must beat current trick (if it's also a bomb).
        if let Some(ref current) = self.current_trick {
            if !can_beat(current, &combo) {
                return Err(GameError::CannotBeatCurrentTrick);
            }
        }

        // Remove cards from hand.
        for c in cards {
            if let Some(pos) = self.hands[seat].iter().position(|h| h == c) {
                self.hands[seat].remove(pos);
            }
        }

        self.current_trick_cards.extend_from_slice(cards);
        self.current_trick = Some(combo);
        self.last_played_by = Some(seat);
        self.pass_count = 0;
        self.first_card_played = true;

        // Check if player went out.
        if self.hands[seat].is_empty() {
            self.out_order.push(seat);
            self.check_wish_fulfillment();
        }

        if self.should_round_end() {
            self.end_round();
            return Ok(());
        }

        // Turn advances to next player after bomber.
        self.current_player = self.get_next_active_seat(seat);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Dragon giveaway
    // -----------------------------------------------------------------------

    /// After winning a trick with Dragon, the winner must give it to an opponent.
    pub fn give_dragon_trick(&mut self, seat: Seat, target: Seat) -> Result<(), GameError> {
        if !self.dragon_giveaway {
            return Err(GameError::WrongPhase);
        }
        if self.dragon_giveaway_seat != Some(seat) {
            return Err(GameError::NotYourTurn);
        }
        // Target must be on opposing team.
        if team_for_seat(target) == team_for_seat(seat) {
            return Err(GameError::InvalidDragonTarget);
        }

        // Give trick cards to target.
        let trick_cards = std::mem::take(&mut self.current_trick_cards);
        self.tricks_won[target].push(trick_cards);

        self.dragon_giveaway = false;
        self.dragon_giveaway_seat = None;
        self.current_trick = None;

        // Next player leads.
        if self.is_out(seat) {
            self.current_player = self.get_next_active_seat(seat);
        } else {
            self.current_player = seat; // Winner leads.
        }

        if self.should_round_end() {
            self.end_round();
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Trick resolution
    // -----------------------------------------------------------------------

    fn award_trick(&mut self) {
        let winner = self.last_played_by.unwrap();

        // Check if dragon is in the trick.
        let has_dragon = self
            .current_trick_cards
            .iter()
            .any(|c| matches!(c, Card::Special(SpecialKind::Dragon)));

        if has_dragon {
            // Dragon winner must give trick to an opponent.
            self.dragon_giveaway = true;
            self.dragon_giveaway_seat = Some(winner);
            // Don't clear trick yet - it will be given away.
            // But clear the combo so no one needs to beat it.
            self.current_trick = None;
            self.pass_count = 0;
            return;
        }

        // Normal trick: cards go to winner.
        let trick_cards = std::mem::take(&mut self.current_trick_cards);
        self.tricks_won[winner].push(trick_cards);
        self.current_trick = None;
        self.pass_count = 0;

        // Winner leads next (or next active if winner is out).
        if self.is_out(winner) {
            self.current_player = self.get_next_active_seat(winner);
        } else {
            self.current_player = winner;
        }
    }

    // -----------------------------------------------------------------------
    // Round end
    // -----------------------------------------------------------------------

    fn should_round_end(&self) -> bool {
        let out_count = self.out_order.len();
        if out_count >= 3 {
            return true;
        }
        // Double victory: first two out are same team.
        if out_count >= 2 && team_for_seat(self.out_order[0]) == team_for_seat(self.out_order[1]) {
            return true;
        }
        false
    }

    fn end_round(&mut self) {
        // If a trick is in progress (not dragon giveaway), award it.
        if !self.dragon_giveaway && self.last_played_by.is_some() && !self.current_trick_cards.is_empty() {
            let winner = self.last_played_by.unwrap();
            let trick_cards = std::mem::take(&mut self.current_trick_cards);
            self.tricks_won[winner].push(trick_cards);
        }

        // Fill remaining out_order slots.
        for seat in 0..4 {
            if !self.out_order.contains(&seat) {
                self.out_order.push(seat);
            }
        }

        let (t0, t1) = score_round(
            &self.out_order,
            &self.tricks_won,
            &self.hands,
            &self.tichu_calls,
        );

        self.scores[0] += t0;
        self.scores[1] += t1;
        self.round_history.push((t0, t1));

        // Check game end.
        if self.is_game_over() {
            self.phase = Phase::GameEnd;
        } else {
            self.phase = Phase::RoundEnd;
        }
    }

    fn is_game_over(&self) -> bool {
        let s0 = self.scores[0];
        let s1 = self.scores[1];
        (s0 >= self.target_score || s1 >= self.target_score) && s0 != s1
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    pub fn is_out(&self, seat: Seat) -> bool {
        self.out_order.contains(&seat)
    }

    fn get_next_active_seat(&self, from: Seat) -> Seat {
        let step = match self.direction {
            Direction::Clockwise => 1,
            Direction::CounterClockwise => 3,
        };
        let mut next = (from + step) % 4;
        for _ in 0..4 {
            if !self.is_out(next) {
                return next;
            }
            next = (next + step) % 4;
        }
        from // shouldn't happen
    }

    fn check_wish_fulfillment(&mut self) {
        if let Some(wish_rank) = self.mah_jong_wish {
            // Wish is fulfilled when no active player holds a card of that rank.
            let anyone_has_it = (0..4)
                .filter(|&s| !self.is_out(s))
                .any(|s| {
                    self.hands[s].iter().any(|c| match c {
                        Card::Normal { rank, .. } => *rank == wish_rank,
                        _ => false,
                    })
                });
            if !anyone_has_it {
                self.mah_jong_wish = None;
            }
        }
    }

    /// Get all legal actions for the current player.
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.phase != Phase::Playing {
            return vec![];
        }

        if self.dragon_giveaway {
            if let Some(seat) = self.dragon_giveaway_seat {
                let mut actions = vec![];
                let left = left_seat(seat);
                let right = right_seat(seat);
                if team_for_seat(left) != team_for_seat(seat) {
                    actions.push(Action::GiveDragonTrick(left));
                }
                if team_for_seat(right) != team_for_seat(seat) {
                    actions.push(Action::GiveDragonTrick(right));
                }
                return actions;
            }
            return vec![];
        }

        let seat = self.current_player;
        let hand = &self.hands[seat];
        let mut actions = Vec::new();

        // Pass (if not leading).
        if self.current_trick.is_some() {
            // Can pass unless wish enforcement prevents it.
            if let Some(wish_rank) = self.mah_jong_wish {
                if !can_fulfill_wish(hand, wish_rank, self.current_trick.as_ref()) {
                    actions.push(Action::Pass);
                }
            } else {
                actions.push(Action::Pass);
            }
        }

        // Playable combos.
        let combos = find_playable_combos(hand, self.current_trick.as_ref());
        for combo in combos {
            // Filter by wish if active.
            if let Some(wish_rank) = self.mah_jong_wish {
                let has_wished = hand.iter().any(|c| match c {
                    Card::Normal { rank, .. } => *rank == wish_rank,
                    _ => false,
                });
                if has_wished && can_fulfill_wish(hand, wish_rank, self.current_trick.as_ref()) {
                    // Must include wished rank (unless bomb).
                    let plays_wish = combo.cards.iter().any(|c| match c {
                        Card::Normal { rank, .. } => *rank == wish_rank,
                        _ => false,
                    });
                    if !plays_wish && !is_bomb_combo(&combo) {
                        continue;
                    }
                }
            }
            actions.push(Action::Play(combo));
        }

        actions
    }

    /// Get the winning team (if game is over).
    pub fn winner(&self) -> Option<Team> {
        if self.phase != Phase::GameEnd {
            return None;
        }
        if self.scores[0] > self.scores[1] {
            Some(0)
        } else if self.scores[1] > self.scores[0] {
            Some(1)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Pass,
    Play(Combo),
    GiveDragonTrick(Seat),
}

fn is_bomb_combo(combo: &Combo) -> bool {
    matches!(
        combo.kind,
        ComboKind::FourOfAKindBomb | ComboKind::StraightFlushBomb
    )
}
