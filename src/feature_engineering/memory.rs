mod energy_field;
mod hidden_parameters;
mod masked_possibilities;
#[allow(dead_code)]
pub mod probabilities;
mod relic_nodes;

use crate::rules_engine::action::Action;
use crate::rules_engine::param_ranges::ParamRanges;
use crate::rules_engine::params::{FixedParams, KnownVariableParams};
use crate::rules_engine::state::{Observation, Pos};
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
    pub fn new(param_ranges: &ParamRanges, map_size: [usize; 2]) -> Self {
        let energy_field = EnergyFieldMemory::new(map_size);
        let hidden_parameters = HiddenParametersMemory::new(param_ranges);
        let relic_nodes = RelicNodeMemory::new(map_size);
        Self {
            energy_field,
            hidden_parameters,
            relic_nodes,
        }
    }

    pub fn update(
        &mut self,
        obs: &Observation,
        last_actions: &[Action],
        fixed_params: &FixedParams,
        params: &KnownVariableParams,
    ) {
        self.energy_field.update(obs);
        self.hidden_parameters
            .update(obs, last_actions, fixed_params, params);
        self.relic_nodes.update(obs);
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
