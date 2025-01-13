use crate::feature_engineering::memory::Memory;
use crate::izip_eq;
use crate::rules_engine::env::get_spawn_position;
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{KnownVariableParams, FIXED_PARAMS};
use crate::rules_engine::state::{Observation, Unit};
use itertools::Itertools;
use numpy::ndarray::{
    ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, Zip,
};
use std::iter::Iterator;
use std::sync::LazyLock;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum TemporalSpatialFeature {
    Visible,
    MyUnitCount,
    OppUnitCount,
    MyUnitEnergy,
    OppUnitEnergy,
}

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum NontemporalSpatialFeature {
    // Static features first
    DistanceFromSpawn,
    // I think these we only need to provide these unit features
    // for the current frame
    MyUnitCanMove,
    OppUnitCanMove,
    MyUnitCanSap,
    OppUnitCanSap,
    // Dead units still provide vision, so we include them here
    MyDeadUnitCount,
    OppDeadUnitCount,
    MyUnitEnergyMin,
    OppUnitEnergyMin,
    MyUnitEnergyMax,
    OppUnitEnergyMax,
    // These features represent the state of the map
    // They also don't seem needed for more than the current frame
    Asteroid,
    Nebula,
    TileExplored,
    RelicNode,
    RelicNodeExplored,
    // 1 if known to have points, 0 otherwise
    TileKnownPoints,
    // The estimated points (only if unknown, 0 if known or there's no data)
    TileEstimatedPoints,
    // Whether the tile is known to have points or not
    TilePointsExplored,
    EnergyField,
}

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum TemporalGlobalFeature {
    MyTeamPoints,
    OppTeamPoints,
}

#[derive(Debug, Clone, Copy, EnumIter)]
enum NontemporalGlobalFeature {
    // Visible features
    MyTeamWins = 0,
    OppTeamWins = 3,
    MatchSteps = 6,
    // Known parameters
    UnitMoveCost = 7,
    UnitSapCost = 12,
    UnitSapRange = 13,
    UnitSensorRange = 18,
    // Estimated / inferred features
    UnitSapDropoffFactor = 21,
    NebulaTileVisionReduction = 24,
    NebulaTileEnergyReduction = 28,
    NebulaTileDriftSpeed = 31,
    EnergyNodeDriftSpeed = 35,
    End = 39,
}

// Normalizing constants
const UNIT_COUNT_NORM: f32 = 4.0;
const UNIT_ENERGY_NORM: f32 = FIXED_PARAMS.max_unit_energy as f32;
const UNIT_ENERGY_MIN_BASELINE: f32 = 0.1;
static ENERGY_FIELD_NORM: LazyLock<f32> = LazyLock::new(|| {
    *PARAM_RANGES
        .nebula_tile_energy_reduction
        .iter()
        .max()
        .unwrap() as f32
});
const MANHATTAN_DISTANCE_NORM: f32 =
    (FIXED_PARAMS.map_width + FIXED_PARAMS.map_height) as f32;
const POINTS_NORM: f32 = 200.0;

static UNIT_SAP_COST_MIN: LazyLock<i32> =
    LazyLock::new(|| *PARAM_RANGES.unit_sap_cost.iter().min().unwrap());
static UNIT_SAP_COST_MAX: LazyLock<i32> =
    LazyLock::new(|| *PARAM_RANGES.unit_sap_cost.iter().max().unwrap());

pub fn get_temporal_spatial_feature_count() -> usize {
    TemporalSpatialFeature::COUNT
}

pub fn get_nontemporal_spatial_feature_count() -> usize {
    NontemporalSpatialFeature::COUNT
}

pub fn get_temporal_global_feature_count() -> usize {
    TemporalGlobalFeature::COUNT
}

pub fn get_nontemporal_global_feature_count() -> usize {
    NontemporalGlobalFeature::End as usize
}

/// Writes into spatial_out of shape (teams, s_channels, map_width, map_height) and
/// global_out of shape (teams, g_channels)
pub fn write_obs_arrays(
    mut temporal_spatial_out: ArrayViewMut4<f32>,
    mut nontemporal_spatial_out: ArrayViewMut4<f32>,
    mut temporal_global_out: ArrayViewMut2<f32>,
    mut nontemporal_global_out: ArrayViewMut2<f32>,
    observations: &[Observation],
    memories: &[Memory],
    params: &KnownVariableParams,
) {
    for (
        obs,
        mem,
        temporal_spatial_out,
        nontemporal_spatial_out,
        temporal_global_out,
        nontemporal_global_out,
    ) in izip_eq!(
        observations,
        memories,
        temporal_spatial_out.outer_iter_mut(),
        nontemporal_spatial_out.outer_iter_mut(),
        temporal_global_out.outer_iter_mut(),
        nontemporal_global_out.outer_iter_mut(),
    ) {
        write_temporal_spatial_out(temporal_spatial_out, obs);
        write_nontemporal_spatial_out(
            nontemporal_spatial_out,
            obs,
            mem,
            params,
        );
        write_temporal_global_out(temporal_global_out, obs);
        write_nontemporal_global_out(nontemporal_global_out, obs, mem, params);
    }
}

fn write_temporal_spatial_out(
    mut temporal_spatial_out: ArrayViewMut3<f32>,
    obs: &Observation,
) {
    use TemporalSpatialFeature::*;

    for (sf, mut slice) in TemporalSpatialFeature::iter()
        .zip_eq(temporal_spatial_out.outer_iter_mut())
    {
        match sf {
            Visible => {
                slice.assign(
                    &obs.sensor_mask.map(|v| if *v { 1.0 } else { 0.0 }),
                );
            },
            MyUnitCount => {
                write_alive_unit_counts(slice, obs.get_my_units());
            },
            OppUnitCount => {
                write_alive_unit_counts(slice, obs.get_opp_units());
            },
            MyUnitEnergy => {
                write_unit_energies(slice, obs.get_my_units());
            },
            OppUnitEnergy => {
                write_unit_energies(slice, obs.get_opp_units());
            },
        }
    }
}

fn write_nontemporal_spatial_out(
    mut nontemporal_spatial_out: ArrayViewMut3<f32>,
    obs: &Observation,
    mem: &Memory,
    params: &KnownVariableParams,
) {
    use NontemporalSpatialFeature::*;

    for (sf, mut slice) in NontemporalSpatialFeature::iter()
        .zip_eq(nontemporal_spatial_out.outer_iter_mut())
    {
        match sf {
            DistanceFromSpawn => {
                let spawn_pos =
                    get_spawn_position(obs.team_id, FIXED_PARAMS.map_size);
                slice.indexed_iter_mut().for_each(|(xy, out)| {
                    *out = spawn_pos.manhattan_distance(xy.into()) as f32
                        / MANHATTAN_DISTANCE_NORM;
                });
            },
            MyUnitCanMove => {
                write_units_have_enough_energy_counts(
                    slice,
                    obs.get_my_units(),
                    params.unit_move_cost,
                );
            },
            OppUnitCanMove => {
                write_units_have_enough_energy_counts(
                    slice,
                    obs.get_opp_units(),
                    params.unit_move_cost,
                );
            },
            MyUnitCanSap => {
                write_units_have_enough_energy_counts(
                    slice,
                    obs.get_opp_units(),
                    params.unit_sap_cost,
                );
            },
            OppUnitCanSap => {
                write_units_have_enough_energy_counts(
                    slice,
                    obs.get_opp_units(),
                    params.unit_sap_cost,
                );
            },
            MyDeadUnitCount => {
                write_dead_unit_counts(slice, obs.get_my_units());
            },
            OppDeadUnitCount => {
                write_dead_unit_counts(slice, obs.get_opp_units());
            },
            MyUnitEnergyMin => {
                write_unit_energy_min(slice, obs.get_my_units());
            },
            OppUnitEnergyMin => {
                write_unit_energy_min(slice, obs.get_opp_units());
            },
            MyUnitEnergyMax => {
                write_unit_energy_max(slice, obs.get_my_units());
            },
            OppUnitEnergyMax => {
                write_unit_energy_max(slice, obs.get_opp_units());
            },
            Asteroid => Zip::from(&mut slice)
                .and(mem.get_known_asteroids_map())
                .for_each(|out, &asteroid| {
                    *out = if asteroid { 1.0 } else { 0.0 }
                }),
            Nebula => Zip::from(&mut slice)
                .and(mem.get_known_nebulae_map())
                .for_each(|out, &nebula| *out = if nebula { 1.0 } else { 0.0 }),
            TileExplored => Zip::from(&mut slice)
                .and(mem.get_explored_tiles_map())
                .for_each(|out, &explored| {
                    *out = if explored { 1.0 } else { 0.0 }
                }),
            RelicNode => {
                mem.get_relic_nodes()
                    .iter()
                    .for_each(|r| slice[r.as_index()] = 1.0);
            },
            RelicNodeExplored => Zip::from(&mut slice)
                .and(mem.get_explored_relic_nodes_map())
                .for_each(|out, &explored| {
                    *out = if explored { 1.0 } else { 0.0 }
                }),
            TileKnownPoints => Zip::from(&mut slice)
                .and(mem.get_relic_known_and_explored_points_map())
                .for_each(|out, &known_and_explored| {
                    *out = if known_and_explored { 1.0 } else { 0.0 }
                }),
            TileEstimatedPoints => {
                slice.assign(mem.get_relic_estimated_points_map())
            },
            TilePointsExplored => Zip::from(&mut slice)
                .and(mem.get_relic_explored_points_map())
                .for_each(|out, &explored| {
                    *out = if explored { 1.0 } else { 0.0 }
                }),
            EnergyField => {
                // Optimistically estimate nebula tile energy reduction
                let nebula_cost = mem
                    .iter_nebula_tile_energy_reduction_options()
                    .copied()
                    .min()
                    .unwrap();
                Zip::from(&mut slice)
                    .and(mem.get_energy_field())
                    .and(mem.get_known_nebulae_map())
                    .for_each(|out, &energy, &is_nebula| {
                        if let Some(e) = energy {
                            let e = if is_nebula { e - nebula_cost } else { e };
                            *out = e as f32 / *ENERGY_FIELD_NORM
                        }
                    })
            },
        }
    }
}

fn write_temporal_global_out(
    mut temporal_global_out: ArrayViewMut1<f32>,
    obs: &Observation,
) {
    use TemporalGlobalFeature::*;

    for (gf, out) in
        TemporalGlobalFeature::iter().zip_eq(temporal_global_out.iter_mut())
    {
        match gf {
            MyTeamPoints => {
                *out = obs.team_points[obs.team_id] as f32 / POINTS_NORM;
            },
            OppTeamPoints => {
                *out = obs.team_points[obs.opp_team_id()] as f32 / POINTS_NORM;
            },
        }
    }
}

fn write_nontemporal_global_out(
    mut nontemporal_global_out: ArrayViewMut1<f32>,
    obs: &Observation,
    mem: &Memory,
    params: &KnownVariableParams,
) {
    use NontemporalGlobalFeature::*;

    let mut global_result: Vec<f32> = vec![0.0; End as usize];
    for (gf, next_gf) in NontemporalGlobalFeature::iter().tuple_windows() {
        match gf {
            MyTeamWins => {
                let my_team_wins =
                    discretize_team_wins(obs.team_wins[obs.team_id]);
                global_result[gf as usize..next_gf as usize]
                    .copy_from_slice(&my_team_wins);
            },
            OppTeamWins => {
                let opp_team_wins =
                    discretize_team_wins(obs.team_wins[obs.opp_team_id()]);
                global_result[gf as usize..next_gf as usize]
                    .copy_from_slice(&opp_team_wins);
            },
            MatchSteps => {
                global_result[gf as usize] = obs.match_steps as f32
                    / FIXED_PARAMS.max_steps_in_match as f32;
            },
            UnitMoveCost => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &one_hot_encode_param_range(
                        params.unit_move_cost,
                        &PARAM_RANGES.unit_move_cost,
                    ),
                );
            },
            UnitSapCost => {
                global_result[gf as usize] =
                    (params.unit_sap_cost - *UNIT_SAP_COST_MIN) as f32
                        / (*UNIT_SAP_COST_MAX - *UNIT_SAP_COST_MIN) as f32;
            },
            UnitSapRange => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &one_hot_encode_param_range(
                        params.unit_sap_range,
                        &PARAM_RANGES.unit_sap_range,
                    ),
                );
            },
            UnitSensorRange => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &one_hot_encode_param_range(
                        params.unit_sensor_range,
                        &PARAM_RANGES.unit_sensor_range,
                    ),
                );
            },
            UnitSapDropoffFactor => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.get_unit_sap_dropoff_factor_weights(),
                );
            },
            NebulaTileVisionReduction => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.get_nebula_tile_vision_reduction_weights(),
                );
            },
            NebulaTileEnergyReduction => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.get_nebula_tile_energy_reduction_weights(),
                );
            },
            NebulaTileDriftSpeed => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.get_nebula_tile_drift_speed_weights(),
                );
            },
            EnergyNodeDriftSpeed => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.get_energy_node_drift_speed_weights(),
                );
            },
            End => {
                unreachable!()
            },
        }
    }
    nontemporal_global_out
        .iter_mut()
        .zip_eq(global_result)
        .for_each(|(out, v)| *out = v);
}

fn write_alive_unit_counts(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units
        .iter()
        .filter(|u| u.alive())
        .for_each(|u| slice[u.pos.as_index()] += 1. / UNIT_COUNT_NORM);
}

fn write_unit_energies(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units.iter().filter(|u| u.alive()).for_each(|u| {
        slice[u.pos.as_index()] += u.energy as f32 / UNIT_ENERGY_NORM
    });
}

fn write_dead_unit_counts(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units
        .iter()
        .filter(|u| !u.alive())
        .for_each(|u| slice[u.pos.as_index()] += 1. / UNIT_COUNT_NORM);
}

fn write_units_have_enough_energy_counts(
    mut slice: ArrayViewMut2<f32>,
    units: &[Unit],
    energy: i32,
) {
    units
        .iter()
        .filter(|u| u.energy >= energy)
        .for_each(|u| slice[u.pos.as_index()] += 1. / UNIT_COUNT_NORM);
}

fn write_unit_energy_min(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units.iter().filter(|u| u.alive()).for_each(|u| {
        let cur_val = slice[u.pos.as_index()];
        let new_val =
            u.energy as f32 / UNIT_ENERGY_NORM + UNIT_ENERGY_MIN_BASELINE;
        slice[u.pos.as_index()] = if cur_val == 0.0 {
            new_val
        } else {
            cur_val.min(new_val)
        }
    });
}

fn write_unit_energy_max(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units.iter().filter(|u| u.alive()).for_each(|u| {
        slice[u.pos.as_index()] =
            slice[u.pos.as_index()].max(u.energy as f32 / UNIT_ENERGY_NORM);
    });
}

fn discretize_team_wins(wins: u32) -> [f32; 3] {
    match wins {
        0 => [1., 0., 0.],
        1 => [0., 1., 0.],
        2.. => [0., 0., 1.],
    }
}

fn one_hot_encode_param_range<T>(val: T, range: &[T]) -> Vec<f32>
where
    T: Copy + Eq,
{
    let mut encoded = vec![0.0; range.len()];
    encoded[range.iter().position(|&v| v == val).unwrap()] = 1.0;
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::param_ranges::{
        IRRELEVANT_ENERGY_NODE_DRIFT_SPEED, PARAM_RANGES,
    };

    #[test]
    fn test_nontemporal_global_feature_indices() {
        use NontemporalGlobalFeature::*;

        for (feature, next_feature) in
            NontemporalGlobalFeature::iter().tuple_windows()
        {
            match feature {
                MyTeamWins | OppTeamWins => {
                    let option_count = discretize_team_wins(0).len();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                UnitMoveCost => {
                    let option_count = PARAM_RANGES
                        .unit_move_cost
                        .iter()
                        .sorted()
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                UnitSapRange => {
                    let option_count = PARAM_RANGES
                        .unit_sap_range
                        .iter()
                        .sorted()
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                UnitSensorRange => {
                    let option_count = PARAM_RANGES
                        .unit_sensor_range
                        .iter()
                        .sorted()
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                UnitSapDropoffFactor => {
                    let option_count = PARAM_RANGES
                        .unit_sap_dropoff_factor
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                NebulaTileVisionReduction => {
                    let option_count = PARAM_RANGES
                        .nebula_tile_vision_reduction
                        .iter()
                        .sorted()
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                NebulaTileEnergyReduction => {
                    let option_count = PARAM_RANGES
                        .nebula_tile_energy_reduction
                        .iter()
                        .sorted()
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                NebulaTileDriftSpeed => {
                    let option_count = PARAM_RANGES
                        .nebula_tile_drift_speed
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                EnergyNodeDriftSpeed => {
                    let option_count = PARAM_RANGES
                        .energy_node_drift_speed
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .filter(|&&speed| {
                            speed != IRRELEVANT_ENERGY_NODE_DRIFT_SPEED
                        })
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                End => panic!("End should be the last feature"),
                _ => {
                    assert_eq!(feature as isize, next_feature as isize - 1)
                },
            }
        }
    }

    #[test]
    fn test_discretize_team_wins() {
        for wins in 0..=5 {
            let mut expected = [0.; 3];
            expected[wins.min(2)] = 1.;
            assert_eq!(discretize_team_wins(wins as u32), expected);
        }
    }
}
