mod energy_field;
mod hidden_parameters;
mod masked_possibilities;
pub mod probabilities;
mod relic_nodes;
mod utils;

use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::FixedParams;
use crate::rules_engine::state::Pos;
use energy_field::EnergyFieldMemory;
use hidden_parameters::HiddenParametersMemory;
use numpy::ndarray::Array2;
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

    pub fn get_energy_field(&self) -> &Array2<Option<i32>> {
        &self.energy_field.energy_field
    }

    pub fn get_nebula_tile_vision_reduction_weights(&self) -> Vec<f32> {
        self.hidden_parameters
            .nebula_tile_vision_reduction
            .get_weighted_possibilities()
    }

    pub fn get_nebula_tile_energy_reduction_weights(&self) -> Vec<f32> {
        self.hidden_parameters
            .nebula_tile_energy_reduction
            .get_weighted_possibilities()
    }

    pub fn get_unit_sap_dropoff_factor_weights(&self) -> Vec<f32> {
        self.hidden_parameters
            .unit_sap_dropoff_factor
            .get_weighted_possibilities()
    }

    pub fn get_relic_nodes(&self) -> &[Pos] {
        &self.relic_nodes.relic_nodes
    }

    pub fn get_explored_relic_nodes_map(&self) -> &Array2<bool> {
        &self.relic_nodes.explored_nodes_map
    }

    pub fn get_relic_points_map(&self) -> &Array2<f32> {
        &self.relic_nodes.points_map
    }

    pub fn get_known_relic_points_map(&self) -> &Array2<bool> {
        &self.relic_nodes.known_points_map
    }
}

// TODO: Track asteroid/nebula movement?
