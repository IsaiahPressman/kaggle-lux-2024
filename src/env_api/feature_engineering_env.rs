use crate::env_api::env_data::{
    ActionInfoArraysView, ObsArraysView, PlayerData,
};
use crate::env_api::utils::{
    player_action_array_to_vec, update_memories_and_write_output_arrays,
};
use crate::feature_engineering::obs_space::basic_obs_space::{
    get_global_feature_count, get_spatial_feature_count,
};
use crate::rules_engine::action::Action;
use crate::rules_engine::params::{KnownVariableParams, FIXED_PARAMS};
use crate::rules_engine::state::LuxPlayerObservation;
use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use strum::EnumCount;

type PyEnvOutputs<'py> = (
    (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<f32>>),
    (
        Bound<'py, PyArray3<bool>>,
        Bound<'py, PyArray4<bool>>,
        Bound<'py, PyArray3<isize>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
    ),
);

#[pyclass]
pub struct FeatureEngineeringEnv {
    player_data: PlayerData,
    team_id: usize,
}

#[pymethods]
impl FeatureEngineeringEnv {
    #[new]
    fn new(team_id: usize, env_params: Bound<'_, PyDict>) -> PyResult<Self> {
        let env_params = env_params.as_any();
        let unit_move_cost =
            env_params.get_item("unit_move_cost")?.extract()?;
        let unit_sap_cost = env_params.get_item("unit_sap_cost")?.extract()?;
        let unit_sap_range =
            env_params.get_item("unit_sap_range")?.extract()?;
        let unit_sensor_range =
            env_params.get_item("unit_sensor_range")?.extract()?;
        let params = KnownVariableParams {
            unit_move_cost,
            unit_sap_cost,
            unit_sap_range,
            unit_sensor_range,
        };
        let player_data = PlayerData::from_player_count_known_params(1, params);
        Ok(Self {
            team_id,
            player_data,
        })
    }

    fn get_empty_outputs<'py>(&self, py: Python<'py>) -> PyEnvOutputs<'py> {
        EnvOutputs::new().into_pyarray_bound(py)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        lux_obs: &str,
        last_actions: PyReadonlyArray2<'py, isize>,
    ) -> PyEnvOutputs<'py> {
        let player_obs: LuxPlayerObservation =
            serde_json::from_str(lux_obs).unwrap();
        let obs =
            player_obs.get_observation(self.team_id, FIXED_PARAMS.map_size);
        let last_actions = player_action_array_to_vec(last_actions.as_array());
        let mut out = EnvOutputs::new();
        let obs_view = ObsArraysView {
            spatial_obs: out.spatial_obs.view_mut(),
            global_obs: out.global_obs.view_mut(),
        };
        let action_info_view = ActionInfoArraysView {
            action_mask: out.action_mask.view_mut(),
            sap_mask: out.sap_mask.view_mut(),
            unit_indices: out.unit_indices.view_mut(),
            unit_energies: out.unit_energies.view_mut(),
            units_mask: out.units_mask.view_mut(),
        };
        update_memories_and_write_output_arrays(
            obs_view,
            action_info_view,
            &mut self.player_data.memories,
            &[obs],
            &[last_actions],
            &self.player_data.known_params,
        );
        out.into_pyarray_bound(py)
    }
}

struct EnvOutputs {
    spatial_obs: Array4<f32>,
    global_obs: Array2<f32>,
    action_mask: Array3<bool>,
    sap_mask: Array4<bool>,
    unit_indices: Array3<isize>,
    unit_energies: Array2<f32>,
    units_mask: Array2<bool>,
}

impl EnvOutputs {
    fn new() -> Self {
        let spatial_obs = Array4::zeros((
            1,
            get_spatial_feature_count(),
            FIXED_PARAMS.map_width,
            FIXED_PARAMS.map_height,
        ));
        let global_obs = Array2::zeros((1, get_global_feature_count()));
        let action_mask =
            Array3::default((1, FIXED_PARAMS.max_units, Action::COUNT));
        let sap_mask = Array4::default((
            1,
            FIXED_PARAMS.max_units,
            FIXED_PARAMS.map_width,
            FIXED_PARAMS.map_height,
        ));
        let unit_indices = Array3::zeros((1, FIXED_PARAMS.max_units, 2));
        let unit_energies = Array2::zeros((1, FIXED_PARAMS.max_units));
        let units_mask = Array2::default((1, FIXED_PARAMS.max_units));
        Self {
            spatial_obs,
            global_obs,
            action_mask,
            sap_mask,
            unit_indices,
            unit_energies,
            units_mask,
        }
    }

    fn into_pyarray_bound(self, py: Python) -> PyEnvOutputs {
        let obs = (
            self.spatial_obs.into_pyarray_bound(py),
            self.global_obs.into_pyarray_bound(py),
        );
        let action_info = (
            self.action_mask.into_pyarray_bound(py),
            self.sap_mask.into_pyarray_bound(py),
            self.unit_indices.into_pyarray_bound(py),
            self.unit_energies.into_pyarray_bound(py),
            self.units_mask.into_pyarray_bound(py),
        );
        (obs, action_info)
    }
}
