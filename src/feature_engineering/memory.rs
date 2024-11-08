mod hidden_parameters;
pub mod probabilities;
mod relic_nodes;

use crate::feature_engineering::memory::probabilities::Probabilities;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::FixedParams;
use hidden_parameters::HiddenParametersMemory;
use relic_nodes::RelicNodeMemory;

pub struct Memory {
    hidden_parameters: HiddenParametersMemory,
    relic_nodes: RelicNodeMemory,
}

impl Memory {
    pub fn new(
        energy_field_probabilities: Probabilities<i32>,
        fixed_params: &FixedParams,
        param_ranges: &ParamRanges,
    ) -> Self {
        let relic_nodes = RelicNodeMemory::new(fixed_params.map_size);
        let hidden_parameters = HiddenParametersMemory::new(
            param_ranges,
            energy_field_probabilities,
            fixed_params.max_units,
        );
        Self {
            hidden_parameters,
            relic_nodes,
        }
    }
}

// TODO: Track asteroid/nebula movement?
// TODO: Track energy field - after getting learning up and running?
