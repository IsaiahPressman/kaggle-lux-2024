use crate::rules_engine::env::estimate_vision_power_map;
use crate::rules_engine::state::Observation;
use numpy::ndarray::Zip;

#[derive(Debug)]
pub struct HiddenParameterMemory {
    pub nebula_tile_vision_reduction: Option<i32>,
    pub nebula_tile_energy_reduction: Option<u32>,
    pub unit_sap_dropoff_factor: Option<f32>,
    pub unit_energy_void_factor: Option<f32>,
    nebula_tile_vision_reduction_options: Vec<i32>,
    nebula_tile_energy_reduction_options: Vec<u32>,
    unit_sap_dropoff_factor_options: Vec<f32>,
    unit_energy_void_factor_options: Vec<f32>,
}

impl HiddenParameterMemory {
    pub fn new(
        nebula_tile_vision_reduction_options: Vec<i32>,
        nebula_tile_energy_reduction_options: Vec<u32>,
        unit_sap_dropoff_factor_options: Vec<f32>,
        unit_energy_void_factor_options: Vec<f32>,
    ) -> Self {
        HiddenParameterMemory {
            nebula_tile_vision_reduction: None,
            nebula_tile_energy_reduction: None,
            unit_sap_dropoff_factor: None,
            unit_energy_void_factor: None,
            nebula_tile_vision_reduction_options,
            nebula_tile_energy_reduction_options,
            unit_sap_dropoff_factor_options,
            unit_energy_void_factor_options,
        }
    }
}

fn determine_nebula_tile_vision_reduction(
    nebula_tile_vision_reduction: &mut Option<i32>,
    nebula_tile_vision_reduction_options: &mut Vec<i32>,
    obs: &Observation,
    map_size: [usize; 2],
    unit_sensor_range: usize,
) {
    if nebula_tile_vision_reduction.is_some() {
        return;
    }

    let expected_vision_power_map = estimate_vision_power_map(
        &obs.units[obs.team_id],
        map_size,
        unit_sensor_range,
    );
    Zip::from(&expected_vision_power_map)
        .and(&obs.sensor_mask)
        .for_each(|expected_vision, can_see| {
            if *expected_vision > 0 && !can_see {
                nebula_tile_vision_reduction_options
                    .retain(|reduction| reduction >= expected_vision)
            }
        });

    if nebula_tile_vision_reduction_options.len() == 1 {
        *nebula_tile_vision_reduction =
            Some(nebula_tile_vision_reduction_options[0])
    } else if nebula_tile_vision_reduction_options.is_empty() {
        panic!("nebula_tile_vision_reduction_options is empty")
    }
}
