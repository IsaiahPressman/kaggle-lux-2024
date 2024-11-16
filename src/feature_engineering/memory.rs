mod energy_field;
mod hidden_parameters;
mod masked_possibilities;
pub mod probabilities;
mod relic_nodes;
mod utils;

use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::FixedParams;
use energy_field::EnergyFieldMemory;
use hidden_parameters::HiddenParametersMemory;
use relic_nodes::RelicNodeMemory;

pub struct Memory {
    energy_field: EnergyFieldMemory,
    hidden_parameters: HiddenParametersMemory,
    relic_nodes: RelicNodeMemory,
}

impl Memory {
    pub fn new(fixed_params: &FixedParams, param_ranges: &ParamRanges) -> Self {
        let energy_field = EnergyFieldMemory::new(fixed_params.map_size);
        let hidden_parameters = HiddenParametersMemory::new(param_ranges);
        let relic_nodes = RelicNodeMemory::new(fixed_params.map_size);
        Self {
            energy_field,
            hidden_parameters,
            relic_nodes,
        }
    }
}

// TODO: Track asteroid/nebula movement?
