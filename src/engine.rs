//! The game state machine. A port of `shared/src/engine.ts` (the production
//! engine at tichu.squidbox.com): same phases, same turn order, same rule
//! edge cases (Mah Jong wish, Dog, Dragon giveaway, trick countdown during
//! which anyone may bomb). Mutable rather than immutable, for speed.
//!
//! Every accepted action is appended to `log`, so a round can be replayed
//! (see `tests/ts_diff`, which replays Rust games through the TS engine).

use serde::{Deserialize, Serialize};

use crate::combinations::*;
use crate::deck::*;
use crate::scoring::*;
use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameError {
    WrongPhase,
    NotYourTurn,
    AlreadyDecided,
    InvalidCards,
    InvalidCombo,
    CannotBeatCurrentTrick,
    MustFulfillWish,
    CannotPass,
    InvalidSeat,
    InvalidDragonTarget,
    AlreadyCalledTichu,
    DogMustBeLead,
    NotABomb,
    InvalidWish,
    Blocked,
}

impl std::fmt::Display for GameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for GameError {}

/// An action a seat (or the table) takes. Logged in application order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Action {
    GrandTichu { seat: Seat, call: bool },
    SmallTichu { seat: Seat },
    PassCards { seat: Seat, pass: PassInfo },
    /// A play on the seat's turn (includes the Dog and on-turn bombs).
    Play { seat: Seat, cards: Vec<Card>, #[serde(rename = "phoenixAs")] phoenix_as: Option<u8> },
    PassTurn { seat: Seat },
    /// A bomb thrown out of turn (or during the trick countdown).
    Bomb { seat: Seat, cards: Vec<Card> },
    Wish { seat: Seat, rank: Option<u8> },
    /// The trick countdown expired with nobody bombing: award the trick.
    ResolveTrick,
    GiveDragon { seat: Seat, to: Seat },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub hand: Vec<Card>,
    pub tricks_won: Vec<Vec<Card>>,
    pub tichu: TichuCall,
    pub has_played_first_card: bool,
    /// 0 = still in, otherwise 1-based finishing position.
    pub out_order: u8,
    pub grand_decided: bool,
    pub pass: Option<PassInfo>,
}

impl Player {
    fn new() -> Self {
        Player {
            hand: vec![],
            tricks_won: vec![],
            tichu: TichuCall::None,
            has_played_first_card: false,
            out_order: 0,
            grand_decided: false,
            pass: None,
        }
    }
    pub fn is_out(&self) -> bool {
        self.out_order != 0
    }
    pub fn captured_points(&self) -> i32 {
        self.tricks_won.iter().map(|t| total_points(t)).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrickPlay {
    pub seat: Seat,
    pub combo: Combo,
}

/// Who has to act next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pending {
    /// These seats still have to decide on Grand Tichu (simultaneous).
    GrandTichu(Vec<Seat>),
    /// These seats still have to choose their pass (simultaneous).
    Passing(Vec<Seat>),
    /// It's this seat's turn to play or pass.
    Turn(Seat),
    /// This seat played the Mah Jong and must name a wish (or none).
    Wish(Seat),
    /// Everyone passed; the trick goes to this seat unless someone bombs.
    TrickCountdown(Seat),
    /// This seat won a trick with the Dragon and must give it away.
    DragonGiveaway(Seat),
    RoundOver,
    GameOver,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub phase: Phase,
    pub settings: Settings,
    pub players: [Player; 4],
    /// Cumulative team scores.
    pub scores: [i32; 2],
    pub round_number: u32,
    /// The combination to beat (None = the turn player leads).
    pub current_trick: Option<Combo>,
    pub trick_plays: Vec<TrickPlay>,
    pub pass_count: usize,
    pub turn: Seat,
    pub last_played_by: Option<Seat>,
    pub wish: Option<Rank>,
    pub wish_pending: bool,
    pub out_count: u8,
    /// The undealt deck for this round (seat i gets deck[14i..14i+14]).
    pub deck: Vec<Card>,
    /// Some(winner) while the trick countdown runs (bombs still allowed).
    pub trick_countdown: Option<Seat>,
    pub dragon_giveaway_by: Option<Seat>,
    /// Every card played this round, in order (including the Dog).
    pub played_cards: Vec<Card>,
    /// Cards each seat received in the pass, with the seat they came from.
    pub received: [Vec<(Card, Seat)>; 4],
    /// Actions applied this round.
    pub log: Vec<Action>,
    pub round_results: Vec<RoundResult>,
}

impl GameState {
    /// A new game. Call `start_round` to deal.
    pub fn new(settings: Settings) -> Self {
        GameState {
            phase: Phase::RoundEnd,
            settings,
            players: [Player::new(), Player::new(), Player::new(), Player::new()],
            scores: [0, 0],
            round_number: 0,
            current_trick: None,
            trick_plays: vec![],
            pass_count: 0,
            turn: 0,
            last_played_by: None,
            wish: None,
            wish_pending: false,
            out_count: 0,
            deck: vec![],
            trick_countdown: None,
            dragon_giveaway_by: None,
            played_cards: vec![],
            received: Default::default(),
            log: vec![],
            round_results: vec![],
        }
    }

    /// A state in the middle of play, with the given hands and `turn` to act
    /// on an empty table. Everyone counts as having played a card already
    /// (so no Small Tichu). Mostly for tests; see also `from_view`.
    pub fn mid_play(settings: Settings, hands: [Vec<Card>; 4], turn: Seat) -> Self {
        let mut s = GameState::new(settings);
        s.phase = Phase::Playing;
        s.round_number = 1;
        s.turn = turn;
        for (p, mut h) in s.players.iter_mut().zip(hands) {
            sort_hand(&mut h);
            p.hand = h;
            p.grand_decided = true;
            p.has_played_first_card = true;
        }
        s
    }

    /// Rebuild a full game state from one seat's view plus a guess at every
    /// hand (`hands[view.seat]` should be the view's own hand). This is the
    /// "determinization" step of sampling-based search: sample hidden hands
    /// (`PlayerView::sample_hands`), rebuild, then simulate with the engine.
    ///
    /// Captured trick piles aren't in the view, only their point totals, so
    /// each seat's pile is replaced by placeholder cards worth the same points
    /// (only scoring reads piles). Only valid during the playing phase.
    pub fn from_view(view: &crate::view::PlayerView, hands: [Vec<Card>; 4]) -> Self {
        assert_eq!(view.phase, Phase::Playing, "from_view needs the playing phase");
        let mut s = GameState::new(view.settings);
        s.phase = Phase::Playing;
        s.scores = view.scores;
        s.round_number = view.round_number;
        for (seat, mut h) in hands.into_iter().enumerate() {
            let pub_ = &view.players[seat];
            sort_hand(&mut h);
            let p = &mut s.players[seat];
            p.hand = h;
            p.tichu = pub_.tichu;
            p.has_played_first_card = pub_.has_played_first_card;
            p.out_order = pub_.out_order;
            p.grand_decided = true;
            p.tricks_won = vec![placeholder_pile(pub_.captured_points)];
        }
        s.out_count = view.players.iter().filter(|p| p.is_out()).count() as u8;
        s.current_trick = view.current_trick.clone();
        s.trick_plays = view.trick_plays.clone();
        s.pass_count = view.pass_count;
        s.turn = view.turn;
        s.last_played_by = view.last_played_by;
        s.wish = view.wish;
        s.wish_pending = view.wish_pending;
        s.trick_countdown = view.trick_countdown;
        s.dragon_giveaway_by = view.dragon_giveaway_by;
        s.played_cards = view.played_cards.clone();
        s.received[view.seat] = view.received.clone();
        s
    }

    /// `from_view` that also keeps the view's log and my pass, so bots that
    /// read the history (the learned ones) see the same features in a
    /// rollout as in the real game. Rule bots read the log too, so this
    /// changes their rollouts; plain `from_view` keeps pimc as it was.
    pub fn from_view_with_history(view: &crate::view::PlayerView, hands: [Vec<Card>; 4]) -> Self {
        let mut s = Self::from_view(view, hands);
        s.log = view.log.clone();
        s.players[view.seat].pass = view.my_pass;
        s
    }

    // -----------------------------------------------------------------------
    // Round setup
    // -----------------------------------------------------------------------

    /// Deal a new round from `deck` (56 cards; see `deck::shuffled_deck`).
    pub fn start_round(&mut self, deck: Vec<Card>) {
        assert_eq!(deck.len(), 56, "deck must have 56 cards");
        for (i, p) in self.players.iter_mut().enumerate() {
            *p = Player::new();
            p.hand = deck[i * 14..i * 14 + 8].to_vec();
            sort_hand(&mut p.hand);
        }
        self.phase = Phase::GrandTichuWindow;
        self.round_number += 1;
        self.current_trick = None;
        self.trick_plays.clear();
        self.pass_count = 0;
        self.turn = 0;
        self.last_played_by = None;
        self.wish = None;
        self.wish_pending = false;
        self.out_count = 0;
        self.deck = deck;
        self.trick_countdown = None;
        self.dragon_giveaway_by = None;
        self.played_cards.clear();
        self.received = Default::default();
        self.log.clear();
    }

    fn deal_remaining(&mut self) {
        for (i, p) in self.players.iter_mut().enumerate() {
            p.hand.extend_from_slice(&self.deck[i * 14 + 8..i * 14 + 14]);
            sort_hand(&mut p.hand);
        }
        self.phase = Phase::Passing;
    }

    // -----------------------------------------------------------------------
    // Generic dispatch
    // -----------------------------------------------------------------------

    pub fn apply(&mut self, action: &Action) -> Result<(), GameError> {
        match action {
            Action::GrandTichu { seat, call } => self.call_grand_tichu(*seat, *call),
            Action::SmallTichu { seat } => self.call_small_tichu(*seat),
            Action::PassCards { seat, pass } => self.pass_cards(*seat, *pass),
            Action::Play { seat, cards, phoenix_as } => self.play(*seat, cards, *phoenix_as),
            Action::PassTurn { seat } => self.pass_turn(*seat),
            Action::Bomb { seat, cards } => self.bomb(*seat, cards),
            Action::Wish { seat, rank } => self.set_wish(*seat, *rank),
            Action::ResolveTrick => self.resolve_trick(),
            Action::GiveDragon { seat, to } => self.give_dragon(*seat, *to),
        }
    }

    // -----------------------------------------------------------------------
    // Tichu calls
    // -----------------------------------------------------------------------

    pub fn call_grand_tichu(&mut self, seat: Seat, call: bool) -> Result<(), GameError> {
        check_seat(seat)?;
        if self.phase != Phase::GrandTichuWindow {
            return Err(GameError::WrongPhase);
        }
        if self.players[seat].grand_decided {
            return Err(GameError::AlreadyDecided);
        }
        let p = &mut self.players[seat];
        p.grand_decided = true;
        if call {
            p.tichu = TichuCall::Grand;
        }
        self.log.push(Action::GrandTichu { seat, call });
        if self.players.iter().all(|p| p.grand_decided) {
            self.deal_remaining();
        }
        Ok(())
    }

    /// Whether `seat` may call Small Tichu now: during play, before their
    /// first card, without an earlier call, and before anyone has gone out.
    pub fn can_call_small_tichu(&self, seat: Seat) -> bool {
        let p = &self.players[seat];
        self.phase == Phase::Playing && !p.has_played_first_card && p.tichu == TichuCall::None && self.out_count == 0
    }

    pub fn call_small_tichu(&mut self, seat: Seat) -> Result<(), GameError> {
        check_seat(seat)?;
        if self.phase != Phase::Playing {
            return Err(GameError::WrongPhase);
        }
        if self.players[seat].tichu != TichuCall::None {
            return Err(GameError::AlreadyCalledTichu);
        }
        if !self.can_call_small_tichu(seat) {
            return Err(GameError::WrongPhase);
        }
        self.players[seat].tichu = TichuCall::Small;
        self.log.push(Action::SmallTichu { seat });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Passing
    // -----------------------------------------------------------------------

    pub fn pass_cards(&mut self, seat: Seat, pass: PassInfo) -> Result<(), GameError> {
        check_seat(seat)?;
        if self.phase != Phase::Passing {
            return Err(GameError::WrongPhase);
        }
        if self.players[seat].pass.is_some() {
            return Err(GameError::AlreadyDecided);
        }
        if !holds_distinct(&self.players[seat].hand, &pass.cards()) {
            return Err(GameError::InvalidCards);
        }
        self.players[seat].pass = Some(pass);
        self.log.push(Action::PassCards { seat, pass });
        if self.players.iter().all(|p| p.pass.is_some()) {
            self.apply_passes();
        }
        Ok(())
    }

    fn apply_passes(&mut self) {
        let passes: Vec<PassInfo> = self.players.iter().map(|p| p.pass.unwrap()).collect();
        for (seat, pass) in passes.iter().enumerate() {
            for c in pass.cards() {
                remove_card(&mut self.players[seat].hand, c);
            }
        }
        for r in 0..4 {
            let (l, p, rt) = (left_seat(r), partner_seat(r), right_seat(r));
            let got = [(passes[l].right, l), (passes[p].partner, p), (passes[rt].left, rt)];
            for (card, from) in got {
                self.players[r].hand.push(card);
                self.received[r].push((card, from));
            }
        }
        for p in self.players.iter_mut() {
            sort_hand(&mut p.hand);
        }
        self.turn = (0..4).find(|&s| self.players[s].hand.contains(&MAH_JONG)).unwrap_or(0);
        self.phase = Phase::Playing;
    }

    // -----------------------------------------------------------------------
    // Playing
    // -----------------------------------------------------------------------

    /// Who has to act next.
    pub fn pending(&self) -> Pending {
        match self.phase {
            Phase::GrandTichuWindow => {
                Pending::GrandTichu((0..4).filter(|&s| !self.players[s].grand_decided).collect())
            }
            Phase::Passing => Pending::Passing((0..4).filter(|&s| self.players[s].pass.is_none()).collect()),
            Phase::Playing => {
                if let Some(s) = self.dragon_giveaway_by {
                    Pending::DragonGiveaway(s)
                } else if self.wish_pending {
                    Pending::Wish(self.turn)
                } else if let Some(w) = self.trick_countdown {
                    Pending::TrickCountdown(w)
                } else {
                    Pending::Turn(self.turn)
                }
            }
            Phase::RoundEnd => Pending::RoundOver,
            Phase::GameEnd => Pending::GameOver,
        }
    }

    /// Is the table waiting on an on-turn play/pass from `seat`?
    fn is_turn_of(&self, seat: Seat) -> bool {
        self.phase == Phase::Playing
            && self.turn == seat
            && self.dragon_giveaway_by.is_none()
            && !self.wish_pending
            && self.trick_countdown.is_none()
            && !self.players[seat].is_out()
    }

    /// Play `cards` on `seat`'s turn. `phoenix_as` picks the Phoenix's reading
    /// where ambiguous (use the value from a generated `Combo`).
    pub fn play(&mut self, seat: Seat, cards: &[Card], phoenix_as: Option<u8>) -> Result<(), GameError> {
        check_seat(seat)?;
        if self.phase != Phase::Playing {
            return Err(GameError::WrongPhase);
        }
        if self.dragon_giveaway_by.is_some() || self.wish_pending || self.trick_countdown.is_some() {
            return Err(GameError::Blocked);
        }
        if self.turn != seat || self.players[seat].is_out() {
            return Err(GameError::NotYourTurn);
        }
        if !holds_distinct(&self.players[seat].hand, cards) {
            return Err(GameError::InvalidCards);
        }

        if cards.len() == 1 && cards[0] == DOG {
            if self.wish.is_some() && !self.wish_compliant(seat, cards) {
                return Err(GameError::MustFulfillWish);
            }
            return self.play_dog(seat);
        }

        let mut combo = identify_combo(cards, phoenix_as).ok_or(GameError::InvalidCombo)?;
        if combo.kind == ComboKind::Single && cards[0] == PHOENIX {
            if let Some(cur) = &self.current_trick {
                combo.rank = cur.rank + 0.5;
            }
        }
        if let Some(cur) = &self.current_trick {
            if !can_beat(cur, &combo) {
                return Err(GameError::CannotBeatCurrentTrick);
            }
        }
        if self.wish.is_some() && !combo.is_bomb() && !self.wish_compliant(seat, cards) {
            return Err(GameError::MustFulfillWish);
        }

        self.log.push(Action::Play { seat, cards: cards.to_vec(), phoenix_as });
        let p = &mut self.players[seat];
        for c in cards {
            remove_card(&mut p.hand, *c);
        }
        p.has_played_first_card = true;
        if let Some(w) = self.wish {
            if cards.iter().any(|c| c.rank() == Some(w)) {
                self.wish = None;
            }
        }
        let need_wish = cards.contains(&MAH_JONG);
        self.mark_out_if_empty(seat);

        self.trick_plays.push(TrickPlay { seat, combo: combo.clone() });
        self.current_trick = Some(combo);
        self.pass_count = 0;
        self.last_played_by = Some(seat);
        self.wish_pending = need_wish;
        // The turn is held on the Mah Jong player until they name a wish.
        self.turn = if need_wish { seat } else { self.next_active(seat) };
        self.played_cards.extend_from_slice(cards);

        if self.should_round_end() {
            self.end_round_or_await_dragon();
        }
        Ok(())
    }

    fn play_dog(&mut self, seat: Seat) -> Result<(), GameError> {
        if self.current_trick.is_some() {
            return Err(GameError::DogMustBeLead);
        }
        self.log.push(Action::Play { seat, cards: vec![DOG], phoenix_as: None });
        remove_card(&mut self.players[seat].hand, DOG);
        self.players[seat].has_played_first_card = true;
        self.mark_out_if_empty(seat);
        self.played_cards.push(DOG);

        let mut next = partner_seat(seat);
        if self.players[next].is_out() {
            next = self.next_active(next);
        }
        if self.should_round_end() {
            self.end_round();
            return Ok(());
        }
        self.turn = next;
        self.current_trick = None;
        self.trick_plays.clear();
        self.pass_count = 0;
        Ok(())
    }

    pub fn pass_turn(&mut self, seat: Seat) -> Result<(), GameError> {
        check_seat(seat)?;
        if self.phase != Phase::Playing {
            return Err(GameError::WrongPhase);
        }
        if self.turn != seat {
            return Err(GameError::NotYourTurn);
        }
        if self.current_trick.is_none() {
            return Err(GameError::CannotPass);
        }
        if self.wish_pending || self.dragon_giveaway_by.is_some() || self.trick_countdown.is_some() {
            return Err(GameError::Blocked);
        }
        if let Some(w) = self.wish {
            if can_fulfill_wish(&self.players[seat].hand, w, self.current_trick.as_ref()) {
                return Err(GameError::MustFulfillWish);
            }
        }
        self.log.push(Action::PassTurn { seat });
        self.pass_count += 1;
        // Won once every active player other than the last to play has passed
        // (the last player may have gone out on that play).
        let needed = (0..4).filter(|&s| !self.players[s].is_out() && Some(s) != self.last_played_by).count();
        if self.pass_count >= needed {
            self.trick_countdown = self.last_played_by;
        } else {
            self.turn = self.next_active(seat);
        }
        Ok(())
    }

    /// End the trick countdown: the last player to play wins the trick.
    pub fn resolve_trick(&mut self) -> Result<(), GameError> {
        if self.trick_countdown.is_none() {
            return Err(GameError::WrongPhase);
        }
        self.log.push(Action::ResolveTrick);
        self.trick_countdown = None;
        self.win_trick();
        Ok(())
    }

    fn win_trick(&mut self) {
        let winner = self.last_played_by.expect("trick without a player");
        if self.current_trick.as_ref().is_some_and(|t| t.contains(SpecialKind::Dragon)) {
            self.dragon_giveaway_by = Some(winner);
            self.pass_count = 0;
            return;
        }
        let cards = self.take_trick_cards();
        self.players[winner].tricks_won.push(cards);
        self.turn = if self.players[winner].is_out() { self.next_active(winner) } else { winner };
        self.clear_trick();
    }

    pub fn give_dragon(&mut self, seat: Seat, to: Seat) -> Result<(), GameError> {
        check_seat(seat)?;
        check_seat(to)?;
        if self.dragon_giveaway_by != Some(seat) {
            return Err(GameError::WrongPhase);
        }
        if team_for_seat(seat) == team_for_seat(to) {
            return Err(GameError::InvalidDragonTarget);
        }
        self.log.push(Action::GiveDragon { seat, to });
        let cards = self.take_trick_cards();
        self.players[to].tricks_won.push(cards);
        self.turn = if self.players[seat].is_out() { self.next_active(seat) } else { seat };
        self.clear_trick();
        self.dragon_giveaway_by = None;
        if self.should_round_end() {
            self.end_round();
        }
        Ok(())
    }

    pub fn set_wish(&mut self, seat: Seat, rank: Option<Rank>) -> Result<(), GameError> {
        check_seat(seat)?;
        if !self.wish_pending {
            return Err(GameError::WrongPhase);
        }
        if self.turn != seat {
            return Err(GameError::NotYourTurn);
        }
        if rank.is_some_and(|r| !(2..=14).contains(&r)) {
            return Err(GameError::InvalidWish);
        }
        self.log.push(Action::Wish { seat, rank });
        self.wish = rank;
        self.wish_pending = false;
        self.turn = self.next_active(seat);
        Ok(())
    }

    /// Throw a bomb, on turn or not. Allowed on any trick (and during the
    /// trick countdown); on an empty table only by the player to lead.
    pub fn bomb(&mut self, seat: Seat, cards: &[Card]) -> Result<(), GameError> {
        check_seat(seat)?;
        if self.phase != Phase::Playing {
            return Err(GameError::WrongPhase);
        }
        if self.wish_pending || self.dragon_giveaway_by.is_some() {
            return Err(GameError::Blocked);
        }
        if self.players[seat].is_out() {
            return Err(GameError::NotYourTurn);
        }
        if !holds_distinct(&self.players[seat].hand, cards) {
            return Err(GameError::InvalidCards);
        }
        if self.current_trick.is_none() && self.turn != seat {
            return Err(GameError::NotYourTurn);
        }
        let combo = identify_combo(cards, None).ok_or(GameError::InvalidCombo)?;
        if !combo.is_bomb() {
            return Err(GameError::NotABomb);
        }
        if let Some(cur) = &self.current_trick {
            if !can_beat(cur, &combo) {
                return Err(GameError::CannotBeatCurrentTrick);
            }
        }

        self.log.push(Action::Bomb { seat, cards: cards.to_vec() });
        let p = &mut self.players[seat];
        for c in cards {
            remove_card(&mut p.hand, *c);
        }
        p.has_played_first_card = true;
        self.mark_out_if_empty(seat);
        if let Some(w) = self.wish {
            if cards.iter().any(|c| c.rank() == Some(w)) {
                self.wish = None;
            }
        }
        self.trick_plays.push(TrickPlay { seat, combo: combo.clone() });
        self.current_trick = Some(combo);
        self.pass_count = 0;
        self.last_played_by = Some(seat);
        self.turn = self.next_active(seat);
        self.played_cards.extend_from_slice(cards);
        self.trick_countdown = None;
        if self.should_round_end() {
            self.end_round();
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Legal moves
    // -----------------------------------------------------------------------

    /// Every play `seat` may make on their turn right now (empty if it isn't
    /// their turn). Includes the Dog when leading and bombs; applies the wish.
    pub fn legal_plays(&self, seat: Seat) -> Vec<Combo> {
        if !self.is_turn_of(seat) {
            return vec![];
        }
        let hand = &self.players[seat].hand;
        let mut plays = playable_combos(hand, self.current_trick.as_ref());
        if let Some(w) = self.wish {
            if can_fulfill_wish(hand, w, self.current_trick.as_ref()) {
                plays.retain(|c| c.is_bomb() || c.contains_rank(w));
            }
        }
        plays
    }

    /// May `seat` pass on their turn right now?
    pub fn can_pass(&self, seat: Seat) -> bool {
        if !self.is_turn_of(seat) || self.current_trick.is_none() {
            return false;
        }
        match self.wish {
            Some(w) => !can_fulfill_wish(&self.players[seat].hand, w, self.current_trick.as_ref()),
            None => true,
        }
    }

    /// Bombs `seat` could throw right now, in or out of turn.
    pub fn legal_bombs(&self, seat: Seat) -> Vec<Combo> {
        if self.phase != Phase::Playing || self.wish_pending || self.dragon_giveaway_by.is_some() {
            return vec![];
        }
        let p = &self.players[seat];
        if p.is_out() || p.hand.len() < 4 {
            return vec![];
        }
        if self.current_trick.is_none() && self.turn != seat {
            return vec![];
        }
        let mut bombs = playable_combos(&p.hand, self.current_trick.as_ref());
        bombs.retain(|c| c.is_bomb());
        bombs
    }

    /// Would playing `cards` satisfy the wish rule for `seat`?
    fn wish_compliant(&self, seat: Seat, cards: &[Card]) -> bool {
        let Some(w) = self.wish else { return true };
        let hand = &self.players[seat].hand;
        if !hand.iter().any(|c| c.rank() == Some(w)) || cards.iter().any(|c| c.rank() == Some(w)) {
            return true;
        }
        !can_fulfill_wish(hand, w, self.current_trick.as_ref())
    }

    // -----------------------------------------------------------------------
    // Round end
    // -----------------------------------------------------------------------

    fn should_round_end(&self) -> bool {
        if self.out_count >= 3 {
            return true;
        }
        if self.out_count == 2 {
            let out: Vec<Seat> = (0..4).filter(|&s| self.players[s].is_out()).collect();
            return team_for_seat(out[0]) == team_for_seat(out[1]);
        }
        false
    }

    /// End the round, unless the play that ended it left the Dragon on top:
    /// then the giveaway comes first (give_dragon ends the round).
    fn end_round_or_await_dragon(&mut self) {
        let dragon_on_top = self.current_trick.as_ref().is_some_and(|t| t.contains(SpecialKind::Dragon));
        if dragon_on_top && self.last_played_by.is_some() && self.out_count >= 3 {
            self.dragon_giveaway_by = self.last_played_by;
            self.pass_count = 0;
            return;
        }
        self.end_round();
    }

    fn end_round(&mut self) {
        // A trick still on the table goes to whoever played last (to an
        // opponent if the Dragon is on top).
        if !self.trick_plays.is_empty() {
            if let Some(last) = self.last_played_by {
                let dragon_on_top = self.current_trick.as_ref().is_some_and(|t| t.contains(SpecialKind::Dragon));
                let to = if dragon_on_top { right_seat(last) } else { last };
                let cards = self.take_trick_cards();
                self.players[to].tricks_won.push(cards);
                self.current_trick = None;
            }
        }
        let out_order = [0, 1, 2, 3].map(|s| self.players[s].out_order);
        let tricks = [0, 1, 2, 3].map(|s| self.players[s].tricks_won.clone());
        let hands = [0, 1, 2, 3].map(|s| self.players[s].hand.clone());
        let calls = [0, 1, 2, 3].map(|s| self.players[s].tichu);
        let result = score_round(&out_order, &tricks, &hands, &calls);
        self.scores[0] += result.total(0);
        self.scores[1] += result.total(1);
        self.round_results.push(result);
        self.phase =
            if is_game_over(self.scores, self.settings.target_score) { Phase::GameEnd } else { Phase::RoundEnd };
    }

    pub fn last_result(&self) -> Option<&RoundResult> {
        self.round_results.last()
    }

    /// The winning team once the game is over.
    pub fn winner(&self) -> Option<Team> {
        if self.phase != Phase::GameEnd {
            return None;
        }
        Some(if self.scores[0] > self.scores[1] { 0 } else { 1 })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    pub fn is_out(&self, seat: Seat) -> bool {
        self.players[seat].is_out()
    }

    /// The seat after `seat` in play order (ignoring who is out).
    pub fn next_seat(&self, seat: Seat) -> Seat {
        if self.settings.clockwise { left_seat(seat) } else { right_seat(seat) }
    }

    /// The next seat after `from`, in play order, that still holds cards.
    pub fn next_active(&self, from: Seat) -> Seat {
        let mut next = self.next_seat(from);
        for _ in 0..4 {
            if !self.players[next].is_out() {
                return next;
            }
            next = self.next_seat(next);
        }
        next
    }

    fn mark_out_if_empty(&mut self, seat: Seat) {
        if self.players[seat].hand.is_empty() && !self.players[seat].is_out() {
            self.out_count += 1;
            self.players[seat].out_order = self.out_count;
        }
    }

    fn take_trick_cards(&mut self) -> Vec<Card> {
        self.trick_plays.iter().flat_map(|p| p.combo.cards.iter().copied()).collect()
    }

    fn clear_trick(&mut self) {
        self.current_trick = None;
        self.trick_plays.clear();
        self.pass_count = 0;
        self.last_played_by = None;
    }
}

fn check_seat(seat: Seat) -> Result<(), GameError> {
    if seat < 4 { Ok(()) } else { Err(GameError::InvalidSeat) }
}

/// Every card is in `hand` and none is listed twice.
fn holds_distinct(hand: &[Card], cards: &[Card]) -> bool {
    let set = CardSet::from_cards(cards);
    set.len() == cards.len() && cards.iter().all(|c| hand.contains(c))
}

/// Cards worth exactly `points` (a multiple of 5), for `from_view`.
fn placeholder_pile(points: i32) -> Vec<Card> {
    let five = Card::Normal { suit: Suit::Jade, rank: 5 };
    let mut pile = vec![];
    let mut p = points;
    if p < 0 {
        pile.push(PHOENIX);
        p += 25;
    }
    for _ in 0..(p.max(0) / 5) {
        pile.push(five);
    }
    pile
}

fn remove_card(hand: &mut Vec<Card>, card: Card) {
    if let Some(i) = hand.iter().position(|c| *c == card) {
        hand.remove(i);
    }
}
