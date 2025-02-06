use itertools::Itertools;
use numpy::ndarray::Array2;
use crate::rules_engine::action::Action;
use crate::rules_engine::params::{FixedParams, FIXED_PARAMS};
use crate::rules_engine::state::{Observation, Pos, Unit};

#[derive(Debug)]
pub struct ActionMemory {
    pub sapped_positions: Vec<Pos>,
    pub adjacent_sap_count: Array2<u8>,
    units_last_turn: Vec<Unit>,
}

impl ActionMemory {
    pub fn new(map_size: [usize; 2]) -> Self {
        Self {
            sapped_positions: Vec::with_capacity(FIXED_PARAMS.max_units),
            adjacent_sap_count: Array2::zeros(map_size),
            units_last_turn: Vec::with_capacity(FIXED_PARAMS.max_units),
        }
    }

    pub fn update(
        &mut self,
        obs: &Observation,
        last_actions: &[Action],
        fixed_params: &FixedParams,
    ) {
        self.sapped_positions.clear();
        self.adjacent_sap_count.fill(0);
        for unit in self.units_last_turn {
            let Action::Sap(sap_deltas) = last_actions[unit.id] else {
                continue;
            };

            let sap_target_pos = unit.pos.maybe_translate(sap_deltas, fixed_params.map_size).expect("Invalid sap_deltas");
            self.sapped_positions.push(sap_target_pos);
            for adjacent_pos in (-1..=1).cartesian_product(-1..=1).filter_map(|(dx, dy)| {
                if dx == 0 && dy == 0 {
                    None
                } else {
                    sap_target_pos.maybe_translate([dx, dy], fixed_params.map_size)
                }}) {
                self.adjacent_sap_count[adjacent_pos.as_index()] += 1;
            }
        }

        self.units_last_turn.clear();
        self.units_last_turn.extend_from_slice(obs.get_my_units());
    }
}
