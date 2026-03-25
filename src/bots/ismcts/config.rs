use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for the ISMCTS bot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ISMCTSConfig {
    /// Total MCTS iterations per decision.
    pub num_iterations: u32,
    /// UCB1 exploration constant.
    pub exploration: f64,
}

impl Default for ISMCTSConfig {
    fn default() -> Self {
        Self {
            num_iterations: 200,
            exploration: 1.41, // sqrt(2)
        }
    }
}

impl ISMCTSConfig {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, data)
    }
}
