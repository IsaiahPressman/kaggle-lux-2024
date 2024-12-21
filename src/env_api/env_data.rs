use crate::feature_engineering::memory::Memory;
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{KnownVariableParams, FIXED_PARAMS, P};
use numpy::ndarray::{
    ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4,
};

pub struct PlayerData {
    pub memories: [Memory; P],
    pub known_params: KnownVariableParams,
}

impl PlayerData {
    pub fn from_known_params(known_params: KnownVariableParams) -> Self {
        let memories = [
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
            Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size),
        ];
        Self {
            memories,
            known_params,
        }
    }
}

pub struct ObsArraysView<'a> {
    pub spatial_obs: ArrayViewMut4<'a, f32>,
    pub global_obs: ArrayViewMut2<'a, f32>,
}

impl ObsArraysView<'_> {
    pub fn reset(&mut self) {
        self.spatial_obs.fill(0.0);
        self.global_obs.fill(0.0);
    }
}

pub struct ActionInfoArraysView<'a> {
    pub action_mask: ArrayViewMut3<'a, bool>,
    pub sap_mask: ArrayViewMut4<'a, bool>,
    pub unit_indices: ArrayViewMut3<'a, isize>,
    pub unit_energies: ArrayViewMut2<'a, f32>,
    pub units_mask: ArrayViewMut2<'a, bool>,
}

impl ActionInfoArraysView<'_> {
    pub fn reset(&mut self) {
        self.action_mask.fill(false);
        self.sap_mask.fill(false);
        self.unit_indices.fill(0);
        self.unit_energies.fill(0.0);
        self.units_mask.fill(false);
    }
}

pub struct SingleEnvView<'a> {
    pub obs_arrays: ObsArraysView<'a>,
    pub action_info_arrays: ActionInfoArraysView<'a>,
    pub reward: ArrayViewMut1<'a, f32>,
    pub done: &'a mut bool,
}
