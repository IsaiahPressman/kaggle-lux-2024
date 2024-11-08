mod hidden_parameters;
mod probabilities;
mod relic_nodes;

use hidden_parameters::HiddenParametersMemory;
use relic_nodes::RelicNodeMemory;

pub struct Memory {
    hidden_parameters: HiddenParametersMemory,
    relic_nodes: RelicNodeMemory,
}

impl Memory {
    pub fn new(map_size: [usize; 2]) -> Self {
        let relic_nodes = RelicNodeMemory::new(map_size);
        Self {
            // TODO: Don't use default - initialize energy probs correctly
            hidden_parameters: HiddenParametersMemory::default(),
            relic_nodes,
        }
    }
}

// TODO: Track asteroid/nebula movement?
// TODO: Track energy field - after getting learning up and running?
