//! Where do the bots disagree with real players, and who was right?
//!
//! Replays every round with a play log from an `npm run export-rounds` file
//! and, at each on-turn decision with 2+ distinct options (suit-only
//! variants merged, as the net sees them), records:
//!
//! - the human's choice, the net's policy over the options and its top
//!   choice, and h2's choice;
//! - Q: each option's mean round margin (the decider's side) over S guessed
//!   deals consistent with the decider's view, rolled out by ROLLOUT in all
//!   seats, with the same guesses for every option (paired), plus the
//!   standard error of each option's gap to the best one. In two-player
//!   endgames Q is the exact solver value instead.
//!
//!     cargo run --release --example human_disagree -- ROUNDS.jsonl --net PATH
//!         [--samples S] [--rollout BOT] [--out DIR] [--threads T]
//!
//! `--out DIR` writes `decisions.jsonl` (everything above, per decision) and
//! `disagreements.txt` (readable: clear human errors the net avoids, and
//! spots where the human beat the net).

use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;

use tichu_engine::bot::Bot;
use tichu_engine::bots::{self, net::Net};
use tichu_engine::endgame::{self, Decision, Node};
use tichu_engine::engine::{Action, GameState};
use tichu_engine::features::*;
use tichu_engine::humanlog;
use tichu_engine::sim::play_from;
use tichu_engine::types::*;
use tichu_engine::view::PlayerView;

struct Item {
    id: String,
    state: GameState,
    seat: Seat,
    human_play: Option<usize>, // index into legal_plays, None = pass
}

struct Out {
    id: String,
    seat: Seat,
    labels: Vec<String>,
    human: usize,
    net_p: Vec<f32>,
    net_top: usize,
    h2: usize,
    q: Vec<f64>,
    /// Standard error of (Q[best] - Q[k]) per option (0 when exact).
    se_gap: Vec<f64>,
    exact: bool,
    leading: bool,
    cards_left: usize,
    text: String,
}

fn combo_label(c: &Combo) -> String {
    format!("{} [{:?}]", c.cards.iter().map(|x| x.short()).collect::<Vec<_>>().join(" "), c.kind)
}

fn hand_str(h: &[Card]) -> String {
    h.iter().map(|c| c.short()).collect::<Vec<_>>().join(" ")
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn find_play(plays: &[Combo], action: &Action, state: &GameState) -> Option<Option<usize>> {
    match action {
        Action::PassTurn { .. } => Some(None),
        Action::Play { cards, .. } => {
            let set = CardSet::from_cards(cards).0;
            let mut g = state.clone();
            g.apply(action).ok()?;
            let rank = g.trick_plays.last().map(|t| t.combo.rank);
            let cands: Vec<usize> = (0..plays.len()).filter(|&i| CardSet::from_cards(&plays[i].cards).0 == set).collect();
            let pick = cands.iter().copied().find(|&i| Some(plays[i].rank) == rank).or(cands.first().copied())?;
            Some(Some(pick))
        }
        _ => None,
    }
}

fn describe(state: &GameState, view: &PlayerView) -> String {
    let seat = view.seat;
    let mut s = String::new();
    let calls: Vec<String> =
        (0..4).filter(|&x| view.players[x].tichu != TichuCall::None).map(|x| format!("{x}:{:?}", view.players[x].tichu)).collect();
    s += &format!("   hand: {}\n", hand_str(&view.hand));
    let counts: Vec<String> = (0..4)
        .map(|x| {
            let p = &view.players[x];
            let rel = match (x + 4 - seat) % 4 {
                0 => "me",
                1 => "next",
                2 => "partner",
                _ => "prev",
            };
            if p.is_out() { format!("{rel}=out#{}", p.out_order) } else { format!("{rel}={}", p.card_count) }
        })
        .collect();
    let table = match &view.current_trick {
        Some(c) => format!("{} by seat {}", hand_str(&c.cards), view.last_played_by.map(|x| x.to_string()).unwrap_or("?".into())),
        None => "empty (leading)".into(),
    };
    s += &format!(
        "   cards: {} | table: {table} ({} pts) | calls: {} | wish {:?}\n",
        counts.join(" "),
        view.trick_points(),
        if calls.is_empty() { "none".into() } else { calls.join(",") },
        view.wish
    );
    let others: Vec<String> =
        (0..4).filter(|&x| x != seat && !state.players[x].is_out()).map(|x| format!("{x}: {}", hand_str(&state.players[x].hand))).collect();
    s += &format!("   (hidden hands: {})\n", others.join(" | "));
    s
}

fn analyze(it: &Item, net: &Net, rollout: &str, samples: usize, seed: u64) -> Option<Out> {
    let st = &it.state;
    let seat = it.seat;
    let view = st.view(seat);
    let plays = st.legal_plays(seat);
    let can_pass = st.can_pass(seat);
    let cands = Candidates::new(&view, &plays, can_pass);
    if cands.len() < 2 {
        return None;
    }
    let human = cands.option_of(it.human_play)?;
    let labels: Vec<String> = cands.options.iter().map(|o| o.map(|i| combo_label(&plays[i])).unwrap_or("pass".into())).collect();
    // Net policy.
    let mut sb = Vec::with_capacity(STATE_DIM);
    encode_state(&view, &mut sb);
    let net_p = softmax(&net.logits(&net.trunk(&sb), &cands.features));
    let net_top = (0..net_p.len()).max_by(|&a, &b| net_p[a].total_cmp(&net_p[b])).unwrap();
    let h2 = cands.option_of(bots::by_name("h2", 0).unwrap().play(&view, &plays, can_pass))?;
    let team = view.my_team();

    // Q: exact in a two-player endgame, else paired PIMC rollouts.
    let n = cands.len();
    let (q, se_gap, exact) = match endgame::solve_options(st, Node::Engine, 5_000_000) {
        Some((s, opts)) if s == seat => {
            let sign = if team == 0 { 1.0 } else { -1.0 };
            let q: Vec<f64> = cands
                .options
                .iter()
                .map(|o| {
                    let v = opts.iter().find(|ov| match (&ov.decision, o) {
                        (Decision::Pass, None) => true,
                        (Decision::Play(c), Some(i)) => CardSet::from_cards(&c.cards) == CardSet::from_cards(&plays[*i].cards) && c.rank == plays[*i].rank,
                        _ => false,
                    });
                    v.map(|v| sign * v.value as f64).unwrap_or(f64::NAN)
                })
                .collect();
            (q, vec![0.0; n], true)
        }
        _ => {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut m = vec![vec![0.0f64; n]; samples];
            let mut rb: [Box<dyn Bot>; 4] = [0, 1, 2, 3].map(|k| bots::by_name(rollout, seed + k).unwrap());
            for row in m.iter_mut() {
                let base = GameState::from_view(&view, view.sample_hands(&mut rng));
                for (k, o) in cands.options.iter().enumerate() {
                    let mut g = base.clone();
                    let a = match o {
                        Some(i) => Action::Play { seat, cards: plays[*i].cards.clone(), phoenix_as: plays[*i].phoenix_as },
                        None => Action::PassTurn { seat },
                    };
                    g.apply(&a).ok()?;
                    let r = play_from(&mut g, &mut rb, None, false).ok()?;
                    row[k] = (r.total(team) - r.total(1 - team)) as f64;
                }
            }
            let q: Vec<f64> = (0..n).map(|k| m.iter().map(|r| r[k]).sum::<f64>() / samples as f64).collect();
            let best = (0..n).max_by(|&a, &b| q[a].total_cmp(&q[b])).unwrap();
            let se: Vec<f64> = (0..n)
                .map(|k| {
                    let d: Vec<f64> = m.iter().map(|r| r[best] - r[k]).collect();
                    let mu = d.iter().sum::<f64>() / samples as f64;
                    let var = d.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (samples as f64 - 1.0).max(1.0);
                    (var / samples as f64).sqrt()
                })
                .collect();
            (q, se, false)
        }
    };
    Some(Out {
        id: it.id.clone(),
        seat,
        labels,
        human,
        net_p,
        net_top,
        h2,
        q,
        se_gap,
        exact,
        leading: view.current_trick.is_none(),
        cards_left: view.hand.len(),
        text: describe(st, &view),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let path = args.first().expect("usage: human_disagree ROUNDS.jsonl --net PATH");
    let net_path = get("--net").expect("--net PATH");
    let samples: usize = get("--samples").and_then(|s| s.parse().ok()).unwrap_or(48);
    let rollout = get("--rollout").unwrap_or_else(|| format!("net:{net_path}"));
    let threads: usize = get("--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let net = Net::cached(&net_path).expect("load net");

    let rounds = humanlog::read_rounds(&std::fs::read_to_string(path).unwrap());
    let mut items = vec![];
    for r in rounds.iter().filter(|r| !r.plays.is_empty()) {
        let Ok(rp) = humanlog::replay(r) else { continue };
        for (k, stp) in rp.steps.iter().enumerate() {
            if !stp.is_turn_decision() {
                continue;
            }
            let seat = match stp.action {
                Action::Play { seat, .. } | Action::PassTurn { seat } => seat,
                _ => continue,
            };
            let plays = stp.state.legal_plays(seat);
            let Some(hp) = find_play(&plays, &stp.action, &stp.state) else { continue };
            items.push(Item { id: format!("{}.{k}", r.id()), state: stp.state.clone(), seat, human_play: hp });
        }
    }
    eprintln!("{} human turn decisions", items.len());

    let queue = Arc::new(Mutex::new(items.into_iter().enumerate().collect::<Vec<_>>()));
    let results = Arc::new(Mutex::new(vec![]));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let (queue, results, net, rollout) = (queue.clone(), results.clone(), net.clone(), rollout.clone());
            std::thread::spawn(move || loop {
                let Some((i, it)) = queue.lock().unwrap().pop() else { break };
                if let Some(o) = analyze(&it, &net, &rollout, samples, 7 + i as u64) {
                    results.lock().unwrap().push((i, o));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let mut res = Arc::try_unwrap(results).ok().unwrap().into_inner().unwrap();
    res.sort_by_key(|(i, _)| *i);
    let res: Vec<Out> = res.into_iter().map(|(_, o)| o).collect();

    // ---- Summary ----
    let n = res.len() as f64;
    let agree = |f: &dyn Fn(&Out) -> bool| 100.0 * res.iter().filter(|o| f(o)).count() as f64 / n;
    let qbest = |o: &Out| (0..o.q.len()).max_by(|&a, &b| o.q[a].total_cmp(&o.q[b])).unwrap();
    let nll: f64 = res.iter().map(|o| -(o.net_p[o.human].max(1e-6) as f64).ln()).sum::<f64>() / n;
    println!("{} human decisions with 2+ distinct options ({} solved exactly)", res.len(), res.iter().filter(|o| o.exact).count());
    println!("Q = mean round margin over {samples} guessed deals, rollouts by {rollout} (exact in 2-player endgames)\n");
    println!("Agreement with the human's choice (top-1):");
    println!("  net top       {:.1}%   (net log-loss on human moves {:.3})", agree(&|o| o.net_top == o.human), nll);
    println!("  h2            {:.1}%", agree(&|o| o.h2 == o.human));
    println!("  Q argmax      {:.1}%", agree(&|o| qbest(o) == o.human));
    for (name, f) in [
        ("leading", Box::new(|o: &Out| o.leading) as Box<dyn Fn(&Out) -> bool>),
        ("following", Box::new(|o: &Out| !o.leading)),
    ] {
        let sub: Vec<&Out> = res.iter().filter(|o| f(o)).collect();
        let m = sub.len() as f64;
        println!(
            "    {name:<10} n={:<4} net {:.1}%  h2 {:.1}%  Q {:.1}%",
            sub.len(),
            100.0 * sub.iter().filter(|o| o.net_top == o.human).count() as f64 / m,
            100.0 * sub.iter().filter(|o| o.h2 == o.human).count() as f64 / m,
            100.0 * sub.iter().filter(|o| qbest(o) == o.human).count() as f64 / m
        );
    }
    // Average Q loss (vs the best option) of each policy's choices.
    let loss = |pick: &dyn Fn(&Out) -> usize| res.iter().map(|o| o.q[qbest(o)] - o.q[pick(o)]).sum::<f64>() / n;
    println!("\nMean Q shortfall vs the best option, per decision (lower is better; Q argmax is 0 by construction, and biased):");
    println!("  human {:.1} | net {:.1} | h2 {:.1}", loss(&|o| o.human), loss(&|o| o.net_top), loss(&|o| o.h2));

    // A gap counts when it is >= 20 points and >= 2 standard errors (or exact).
    let clear = |o: &Out, a: usize, b: usize| {
        let gap = o.q[a] - o.q[b];
        let se = (o.se_gap[a].powi(2) + o.se_gap[b].powi(2)).sqrt();
        gap >= 20.0 && (o.exact || gap >= 2.0 * se)
    };
    let human_err: Vec<&Out> = res.iter().filter(|o| o.net_top != o.human && clear(o, o.net_top, o.human)).collect();
    let net_err: Vec<&Out> = res.iter().filter(|o| o.net_top != o.human && clear(o, o.human, o.net_top)).collect();
    let disagree = res.iter().filter(|o| o.net_top != o.human).count();
    println!("\nNet and human disagree on {disagree} decisions ({:.1}%):", 100.0 * disagree as f64 / n);
    println!("  human clearly worse (>=20 pts and 2 SE): {}", human_err.len());
    println!("  net clearly worse:                       {}", net_err.len());
    println!("  too close to call:                       {}", disagree - human_err.len() - net_err.len());

    if let Some(dir) = get("--out") {
        std::fs::create_dir_all(&dir).unwrap();
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/decisions.jsonl")).unwrap());
        for o in &res {
            writeln!(
                f,
                "{}",
                json!({"id": o.id, "seat": o.seat, "options": o.labels, "human": o.human, "net_p": o.net_p,
                       "net_top": o.net_top, "h2": o.h2, "q": o.q, "se_gap": o.se_gap, "exact": o.exact,
                       "leading": o.leading, "cards_left": o.cards_left})
            )
            .unwrap();
        }
        let mut t = std::fs::File::create(format!("{dir}/disagreements.txt")).unwrap();
        let show = |t: &mut std::fs::File, title: &str, list: &mut Vec<&Out>, a: &dyn Fn(&Out) -> usize, b: &dyn Fn(&Out) -> usize| {
            list.sort_by(|x, y| (y.q[a(y)] - y.q[b(y)]).total_cmp(&(x.q[a(x)] - x.q[b(x)])));
            writeln!(t, "##### {title}: {}\n", list.len()).unwrap();
            for o in list.iter() {
                writeln!(t, "== {} seat {}{}", o.id, o.seat, if o.exact { " (exact endgame)" } else { "" }).unwrap();
                write!(t, "{}", o.text).unwrap();
                let mut order: Vec<usize> = (0..o.q.len()).collect();
                order.sort_by(|&x, &y| o.q[y].total_cmp(&o.q[x]));
                // The top options, plus the human's, net's and h2's picks wherever they rank.
                let shown: Vec<usize> = order
                    .iter()
                    .enumerate()
                    .filter(|&(r, &k)| r < 6 || k == o.human || k == o.net_top || k == o.h2)
                    .map(|(_, &k)| k)
                    .collect();
                for k in shown {
                    let mut tags = vec![];
                    if k == o.human {
                        tags.push("HUMAN");
                    }
                    if k == o.net_top {
                        tags.push("NET");
                    }
                    if k == o.h2 {
                        tags.push("H2");
                    }
                    writeln!(t, "   Q {:>+7.1} ±{:<5.1} p={:.2}  {:<36} {}", o.q[k], o.se_gap[k], o.net_p[k], o.labels[k], tags.join(" ")).unwrap();
                }
                writeln!(t).unwrap();
            }
        };
        let mut he = human_err.clone();
        show(&mut t, "Human clearly worse than the net", &mut he, &|o| o.net_top, &|o| o.human);
        let mut ne = net_err.clone();
        show(&mut t, "Net clearly worse than the human", &mut ne, &|o| o.human, &|o| o.net_top);
        eprintln!("wrote {dir}/decisions.jsonl and {dir}/disagreements.txt");
    }
}
