/// Tunable weights and thresholds for the heuristic bot.
/// This is the primary target for autoresearch optimization.
#[derive(Debug, Clone)]
pub struct HeuristicWeights {
    // --- Hand evaluation ---
    /// Bonus for holding a four-of-a-kind or straight-flush bomb.
    pub bomb_bonus: f64,
    /// Value of Dragon in hand strength calculation.
    pub dragon_value: f64,
    /// Value of Phoenix in hand strength calculation.
    pub phoenix_value: f64,
    /// Penalty per card group (more groups = more fragmented = weaker).
    pub group_penalty: f64,

    // --- Leading ---
    /// When hand is strong (tempo <= this), lead low to bait opponents.
    pub lead_low_tempo_threshold: f64,

    // --- Following ---
    /// Weight favoring minimum-beat play (1.0 = always min beat, 0.0 = consider stronger).
    pub min_beat_preference: f64,
    /// Only use a bomb if the trick's point value is at least this much.
    pub bomb_point_threshold: f64,

    // --- Tichu calling ---
    /// Hand strength threshold for calling Grand Tichu (8 cards seen).
    pub grand_tichu_threshold: f64,
    /// Tempo threshold for calling Small Tichu (tempo <= this means call).
    pub small_tichu_tempo_threshold: f64,
}

impl Default for HeuristicWeights {
    fn default() -> Self {
        Self {
            bomb_bonus: 8.0,
            dragon_value: 5.0,
            phoenix_value: 4.0,
            group_penalty: 2.0,

            lead_low_tempo_threshold: 6.0,

            min_beat_preference: 0.8,
            bomb_point_threshold: 15.0,

            grand_tichu_threshold: 18.0,
            small_tichu_tempo_threshold: 5.0,
        }
    }
}
