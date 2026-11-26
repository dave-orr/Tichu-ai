//! How many decisions of each kind a round holds, and how many options each
//! has: `cargo run --release --example decision_census -- BOT [ROUNDS]`.
use std::cell::RefCell;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tichu_engine::bot::Bot;
use tichu_engine::bots;
use tichu_engine::deck::shuffled_deck;
use tichu_engine::engine::GameState;
use tichu_engine::features::Candidates;
use tichu_engine::sim::play_dealt_round;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

#[derive(Default)]
struct Census {
    grand: [u64; 2],
    small: [u64; 2],
    pass: u64,
    play: u64,
    play_forced: u64,
    play_options: u64,
    bomb_offers: u64,
    bombs: u64,
    wish: u64,
    dragon: u64,
}

struct Count {
    inner: Box<dyn Bot>,
    c: Rc<RefCell<Census>>,
}

impl Bot for Count {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn grand_tichu(&mut self, v: &PlayerView) -> bool {
        let r = self.inner.grand_tichu(v);
        self.c.borrow_mut().grand[r as usize] += 1;
        r
    }
    fn small_tichu(&mut self, v: &PlayerView) -> bool {
        let r = self.inner.small_tichu(v);
        self.c.borrow_mut().small[r as usize] += 1;
        r
    }
    fn pass_cards(&mut self, v: &PlayerView) -> PassInfo {
        self.c.borrow_mut().pass += 1;
        self.inner.pass_cards(v)
    }
    fn play(&mut self, v: &PlayerView, plays: &[Combo], can_pass: bool) -> Option<usize> {
        let n = Candidates::new(v, plays, can_pass).len();
        let mut c = self.c.borrow_mut();
        if n < 2 {
            c.play_forced += 1;
        } else {
            c.play += 1;
            c.play_options += n as u64;
        }
        drop(c);
        self.inner.play(v, plays, can_pass)
    }
    fn bomb(&mut self, v: &PlayerView, bombs: &[Combo]) -> Option<usize> {
        let r = self.inner.bomb(v, bombs);
        let mut c = self.c.borrow_mut();
        c.bomb_offers += 1;
        c.bombs += r.is_some() as u64;
        r
    }
    fn wish(&mut self, v: &PlayerView) -> Option<Rank> {
        self.c.borrow_mut().wish += 1;
        self.inner.wish(v)
    }
    fn dragon_recipient(&mut self, v: &PlayerView) -> Seat {
        self.c.borrow_mut().dragon += 1;
        self.inner.dragon_recipient(v)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bot = args.get(1).cloned().unwrap_or("h2".into());
    let rounds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let c = Rc::new(RefCell::new(Census::default()));
    let t = std::time::Instant::now();
    for i in 0..rounds {
        let mut bots: [Box<dyn Bot>; 4] =
            [0, 1, 2, 3].map(|s| -> Box<dyn Bot> { Box::new(Count { inner: bots::by_name(&bot, i * 4 + s).expect("bot"), c: c.clone() }) });
        let mut rng = StdRng::seed_from_u64(i);
        let mut state = GameState::new(Settings::default());
        state.start_round(shuffled_deck(&mut rng));
        play_dealt_round(&mut state, &mut bots, None).unwrap();
    }
    let c = c.borrow();
    let r = rounds as f64;
    println!("{bot}: {rounds} rounds, {:.1} ms per round on one thread", t.elapsed().as_secs_f64() * 1000.0 / r);
    println!("per round: grand asked {:.2} (called {:.3}), small offers {:.2} (called {:.3}), passes {:.0}", (c.grand[0] + c.grand[1]) as f64 / r, c.grand[1] as f64 / r, (c.small[0] + c.small[1]) as f64 / r, c.small[1] as f64 / r, c.pass as f64 / r);
    println!("  plays with 2+ options {:.1} (mean options {:.1}), forced {:.1}", c.play as f64 / r, c.play_options as f64 / c.play.max(1) as f64, c.play_forced as f64 / r);
    println!("  bomb offers {:.2} (thrown {:.3}), wishes {:.2}, Dragon gifts {:.2}", c.bomb_offers as f64 / r, c.bombs as f64 / r, c.wish as f64 / r, c.dragon as f64 / r);
}
