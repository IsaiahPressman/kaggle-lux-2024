use crate::feature_engineering::memory::Memory;
use crate::feature_engineering::utils::one_hot_float_encode_param_range;
use crate::izip_eq;
use crate::rules_engine::env::{
    estimate_vision_power_map, get_spawn_position, UNIT_VISION_BONUS,
};
use crate::rules_engine::param_ranges::PARAM_RANGES;
use crate::rules_engine::params::{KnownVariableParams, FIXED_PARAMS};
use crate::rules_engine::state::{Observation, Unit};
use itertools::Itertools;
use numpy::ndarray::{
    s, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, Axis, Zip,
};
use std::iter::Iterator;
use std::sync::LazyLock;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

const FEATURE_BOUND: f32 = 5.0;
const FUTURE_FRAMES: usize = 5;

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum TemporalSpatialFeature {
    Visible,
    EstimatedVisionPower,
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
enum FutureNontemporalSpatialFeature {
    AsteroidFuture,
    NebulaFuture,
}

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum TemporalGlobalFeature {
    MyTeamPoints,
    OppTeamPoints,
    PointsDiff,
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
    UnitSapDropoffFactor = 22,
    NebulaTileVisionReduction = 25,
    NebulaTileEnergyReduction = 33,
    NebulaTileDriftSpeed = 39,
    NebulaTileDriftDirection = 43,
    EnergyNodeDriftSpeed = 45,
    End = 50,
}

// Normalizing constants
static VISION_NORM: LazyLock<f32> = LazyLock::new(|| {
    let max_sensor_range =
        *PARAM_RANGES.unit_sensor_range.iter().max().unwrap();
    let norm = 4 * (max_sensor_range + 1);
    norm as f32
});
const UNIT_COUNT_NORM: f32 = 4.;
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
const POINTS_NORM: f32 = 200.;
const POINTS_DIFF_NORM: f32 = POINTS_NORM / 2.;

static UNIT_SAP_COST_MIN: LazyLock<i32> =
    LazyLock::new(|| *PARAM_RANGES.unit_sap_cost.iter().min().unwrap());
static UNIT_SAP_COST_MAX: LazyLock<i32> =
    LazyLock::new(|| *PARAM_RANGES.unit_sap_cost.iter().max().unwrap());

pub const fn get_temporal_spatial_feature_count() -> usize {
    TemporalSpatialFeature::COUNT
}

pub const fn get_nontemporal_spatial_feature_count() -> usize {
    NontemporalSpatialFeature::COUNT
        + FutureNontemporalSpatialFeature::COUNT * FUTURE_FRAMES
}

pub const fn get_temporal_global_feature_count() -> usize {
    TemporalGlobalFeature::COUNT
}

pub const fn get_nontemporal_global_feature_count() -> usize {
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
        write_temporal_spatial_out(temporal_spatial_out, obs, mem, params);
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
    mem: &Memory,
    params: &KnownVariableParams,
) {
    use TemporalSpatialFeature::*;

    for (sf, mut slice) in TemporalSpatialFeature::iter()
        .zip_eq(temporal_spatial_out.outer_iter_mut())
    {
        match sf {
            Visible => {
                Zip::from(&mut slice).and(&obs.sensor_mask).for_each(
                    |out, &visible| *out = if visible { 1.0 } else { 0.0 },
                );
            },
            EstimatedVisionPower => {
                let mut estimated_vision_power_map = estimate_vision_power_map(
                    obs.get_my_units(),
                    FIXED_PARAMS.map_size,
                    params.unit_sensor_range,
                );
                // Remove unit vision bonus
                obs.get_my_units().iter().for_each(|unit| {
                    estimated_vision_power_map[unit.pos.as_index()] -=
                        UNIT_VISION_BONUS
                });
                // Pessimistically estimate nebula tile vision reduction
                let nebula_vision_cost = mem
                    .iter_unmasked_nebula_tile_vision_reduction_options()
                    .max()
                    .unwrap();
                obs.nebulae.iter().for_each(|nebula| {
                    estimated_vision_power_map[nebula.as_index()] -=
                        nebula_vision_cost
                });
                Zip::from(&mut slice)
                    .and(&estimated_vision_power_map)
                    .and(&obs.sensor_mask)
                    .for_each(|out, &vision_estimate, &visible| {
                        let vision_power =
                            if visible { vision_estimate.max(0) } else { 0 };
                        *out = vision_power as f32 / *VISION_NORM;
                    })
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
    clip_corners(temporal_spatial_out, FIXED_PARAMS.map_size);
}

fn clip_corners(
    mut spatial_out: ArrayViewMut3<f32>,
    [map_height, map_width]: [usize; 2],
) {
    // Corner values often have outliers caused by stacks of units
    for [x, y] in [
        [0, 0],
        [0, 1],
        [1, 0],
        [map_width - 1, map_height - 1],
        [map_width - 1, map_height - 2],
        [map_width - 2, map_height - 1],
    ] {
        spatial_out
            .slice_mut(s![.., x, y])
            .map_inplace(|v| *v = v.clamp(-FEATURE_BOUND, FEATURE_BOUND));
    }
}

fn write_nontemporal_spatial_out(
    mut nontemporal_spatial_out: ArrayViewMut3<f32>,
    obs: &Observation,
    mem: &Memory,
    params: &KnownVariableParams,
) {
    use FutureNontemporalSpatialFeature::*;
    use NontemporalSpatialFeature::*;

    let (mut main_out, mut future_out) = nontemporal_spatial_out
        .view_mut()
        .split_at(Axis(0), NontemporalSpatialFeature::COUNT);
    for (sf, mut slice) in
        NontemporalSpatialFeature::iter().zip_eq(main_out.outer_iter_mut())
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
                    obs.get_my_units(),
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
                .and(mem.get_relic_known_to_have_points_map())
                .for_each(|out, &known_and_explored| {
                    *out = if known_and_explored { 1.0 } else { 0.0 }
                }),
            TileEstimatedPoints => {
                slice.assign(mem.get_relic_estimated_unexplored_points_map())
            },
            TilePointsExplored => Zip::from(&mut slice)
                .and(mem.get_relic_explored_points_map())
                .for_each(|out, &explored| {
                    *out = if explored { 1.0 } else { 0.0 }
                }),
            EnergyField => {
                // Optimistically estimate nebula tile energy reduction
                let nebula_cost = mem
                    .iter_unmasked_nebula_tile_energy_reduction_options()
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

    assert_eq!(future_out.dim().0 % FUTURE_FRAMES, 0);
    for (sf, mut slice) in FutureNontemporalSpatialFeature::iter()
        .zip_eq(future_out.axis_chunks_iter_mut(Axis(0), FUTURE_FRAMES))
    {
        match sf {
            AsteroidFuture => {
                mem.get_future_asteroids(obs.total_steps, FUTURE_FRAMES)
                    .into_iter()
                    .for_each(|(step, pos)| {
                        let [x, y] = pos.as_index();
                        slice[[step, x, y]] = 1.0;
                    });
            },
            NebulaFuture => {
                mem.get_future_nebulae(obs.total_steps, FUTURE_FRAMES)
                    .into_iter()
                    .for_each(|(step, pos)| {
                        let [x, y] = pos.as_index();
                        slice[[step, x, y]] = 1.0;
                    });
            },
        }
    }

    clip_corners(nontemporal_spatial_out, FIXED_PARAMS.map_size);
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
            PointsDiff => {
                *out = (obs.team_points[obs.team_id] as i32
                    - obs.team_points[obs.opp_team_id()] as i32)
                    as f32
                    / POINTS_DIFF_NORM;
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
                    &one_hot_float_encode_param_range(
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
                    &one_hot_float_encode_param_range(
                        params.unit_sap_range,
                        &PARAM_RANGES.unit_sap_range,
                    ),
                );
            },
            UnitSensorRange => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &one_hot_float_encode_param_range(
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
            NebulaTileDriftDirection => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.get_nebula_tile_drift_direction_weights(),
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_engineering::replay::{load_replay, run_replay};
    use crate::rules_engine::param_ranges::PARAM_RANGES;
    use crate::rules_engine::replay::FullReplay;
    use glob::glob;
    use numpy::ndarray::{arr3, Array2, Array4, ArrayViewD, Axis};
    use rstest::rstest;
    use std::cmp::Ordering;
    use std::path::PathBuf;

    const TEST_FEATURE_BOUND: f32 = FEATURE_BOUND * 1.1;

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
                        .map(|s| s.abs())
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .dedup()
                        .count();
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        option_count as isize
                    );
                },
                NebulaTileDriftDirection => {
                    assert_eq!(next_feature as isize - feature as isize, 2,);
                },
                EnergyNodeDriftSpeed => {
                    let option_count = PARAM_RANGES
                        .energy_node_drift_speed
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
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
    fn test_clip_corners() {
        let fill = FEATURE_BOUND + 1.0;
        let mut array = arr3(&[[[fill; 6]; 6], [[-fill; 6]; 6]]);
        clip_corners(array.view_mut(), [6, 6]);
        let expected_array = arr3(&[
            [
                [FEATURE_BOUND, FEATURE_BOUND, fill, fill, fill, fill],
                [FEATURE_BOUND, fill, fill, fill, fill, fill],
                [fill, fill, fill, fill, fill, fill],
                [fill, fill, fill, fill, fill, fill],
                [fill, fill, fill, fill, fill, FEATURE_BOUND],
                [fill, fill, fill, fill, FEATURE_BOUND, FEATURE_BOUND],
            ],
            [
                [-FEATURE_BOUND, -FEATURE_BOUND, -fill, -fill, -fill, -fill],
                [-FEATURE_BOUND, -fill, -fill, -fill, -fill, -fill],
                [-fill, -fill, -fill, -fill, -fill, -fill],
                [-fill, -fill, -fill, -fill, -fill, -fill],
                [-fill, -fill, -fill, -fill, -fill, -FEATURE_BOUND],
                [-fill, -fill, -fill, -fill, -FEATURE_BOUND, -FEATURE_BOUND],
            ],
        ]);
        assert_eq!(array, expected_array);
    }

    #[test]
    fn test_discretize_team_wins() {
        for wins in 0..=5 {
            let mut expected = [0.; 3];
            expected[wins.min(2)] = 1.;
            assert_eq!(discretize_team_wins(wins as u32), expected);
        }
    }

    fn f32_cmp(a: &f32, b: &f32) -> Ordering {
        a.partial_cmp(b).unwrap()
    }

    fn update_ranges(array: ArrayViewD<f32>, ranges: &mut [(f32, f32)]) {
        for (slice, (min_val, max_val)) in
            array.outer_iter().zip_eq(ranges.iter_mut())
        {
            *min_val =
                min_val.min(slice.iter().copied().min_by(f32_cmp).unwrap());
            *max_val =
                max_val.max(slice.iter().copied().max_by(f32_cmp).unwrap());
        }
    }

    type MinMaxRanges = Vec<(f32, f32)>;
    type AllMinMaxRanges =
        (MinMaxRanges, MinMaxRanges, MinMaxRanges, MinMaxRanges);

    fn get_observed_ranges(full_replay: &FullReplay) -> AllMinMaxRanges {
        let params =
            KnownVariableParams::from(full_replay.params.variable.clone());
        let mut memories = [
            [Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size)],
            [Memory::new(&PARAM_RANGES, FIXED_PARAMS.map_size)],
        ];
        let mut temporal_spatial_obs = Array4::zeros((
            1,
            get_temporal_spatial_feature_count(),
            FIXED_PARAMS.map_width,
            FIXED_PARAMS.map_height,
        ));
        let mut nontemporal_spatial_obs = Array4::zeros((
            1,
            get_nontemporal_spatial_feature_count(),
            FIXED_PARAMS.map_width,
            FIXED_PARAMS.map_height,
        ));
        let mut temporal_global_obs =
            Array2::zeros((1, get_temporal_global_feature_count()));
        let mut nontemporal_global_obs =
            Array2::zeros((1, get_nontemporal_global_feature_count()));

        let mut temporal_spatial_ranges =
            vec![(0_f32, 0_f32); get_temporal_spatial_feature_count()];
        let mut nontemporal_spatial_ranges =
            vec![(0_f32, 0_f32); get_nontemporal_spatial_feature_count()];
        let mut temporal_global_ranges =
            vec![(0_f32, 0_f32); get_temporal_global_feature_count()];
        let mut nontemporal_global_ranges =
            vec![(0_f32, 0_f32); get_nontemporal_global_feature_count()];

        for (_state, actions, obs, _next_state) in run_replay(full_replay) {
            for (mem, actions, obs) in
                izip_eq!(memories.iter_mut(), actions, obs)
            {
                temporal_spatial_obs.fill(0.);
                nontemporal_spatial_obs.fill(0.);
                temporal_global_obs.fill(0.);
                nontemporal_global_obs.fill(0.);

                mem[0].update(&obs, &actions, &FIXED_PARAMS, &params);
                write_obs_arrays(
                    temporal_spatial_obs.view_mut(),
                    nontemporal_spatial_obs.view_mut(),
                    temporal_global_obs.view_mut(),
                    nontemporal_global_obs.view_mut(),
                    &[obs],
                    mem,
                    &params,
                );
                update_ranges(
                    temporal_spatial_obs
                        .index_axis(Axis(0), 0)
                        .view()
                        .into_dyn(),
                    &mut temporal_spatial_ranges,
                );
                update_ranges(
                    nontemporal_spatial_obs
                        .index_axis(Axis(0), 0)
                        .view()
                        .into_dyn(),
                    &mut nontemporal_spatial_ranges,
                );
                update_ranges(
                    temporal_global_obs
                        .index_axis(Axis(0), 0)
                        .view()
                        .into_dyn(),
                    &mut temporal_global_ranges,
                );
                update_ranges(
                    nontemporal_global_obs
                        .index_axis(Axis(0), 0)
                        .view()
                        .into_dyn(),
                    &mut nontemporal_global_ranges,
                )
            }
        }

        (
            temporal_spatial_ranges,
            nontemporal_spatial_ranges,
            temporal_global_ranges,
            nontemporal_global_ranges,
        )
    }

    #[rstest]
    #[ignore = "slow"]
    fn test_observation_range_bounds(
        #[files("src/feature_engineering/test_data/*.json")] path: PathBuf,
    ) {
        let full_replay = load_replay(path);
        let (
            temporal_spatial_ranges,
            nontemporal_spatial_ranges,
            temporal_global_ranges,
            nontemporal_global_ranges,
        ) = get_observed_ranges(&full_replay);
        for (feature_scope_id, feature_iterable) in [
            temporal_spatial_ranges.iter().enumerate(),
            nontemporal_spatial_ranges.iter().enumerate(),
            temporal_global_ranges.iter().enumerate(),
            nontemporal_global_ranges.iter().enumerate(),
        ]
        .into_iter()
        .enumerate()
        {
            for (feature_id, &(min_val, max_val)) in feature_iterable {
                println!(
                    "{:?}",
                    (
                        (feature_scope_id, feature_id),
                        (
                            -TEST_FEATURE_BOUND,
                            min_val,
                            max_val,
                            TEST_FEATURE_BOUND
                        ),
                    )
                );
                if min_val >= 0. {
                    assert!(min_val <= 1.);
                    assert!(max_val <= TEST_FEATURE_BOUND);
                } else {
                    assert!(min_val >= -TEST_FEATURE_BOUND);
                    assert!(max_val <= TEST_FEATURE_BOUND);
                }
            }
        }
    }

    fn reduce_range(left: MinMaxRanges, right: MinMaxRanges) -> MinMaxRanges {
        left.into_iter()
            .zip_eq(right)
            .map(|((min_l, max_l), (min_r, max_r))| {
                (min_l.min(min_r), max_l.max(max_r))
            })
            .collect()
    }

    fn reduce_ranges(
        (a1, b1, c1, d1): AllMinMaxRanges,
        (a2, b2, c2, d2): AllMinMaxRanges,
    ) -> AllMinMaxRanges {
        (
            reduce_range(a1, a2),
            reduce_range(b1, b2),
            reduce_range(c1, c2),
            reduce_range(d1, d2),
        )
    }

    #[test]
    #[ignore = "slow"]
    fn test_observation_range_variation() {
        let (
            temporal_spatial_ranges,
            nontemporal_spatial_ranges,
            temporal_global_ranges,
            _nontemporal_global_ranges,
        ) = glob("src/feature_engineering/test_data/*.json")
            .unwrap()
            .map(|path| {
                let full_replay = load_replay(path.unwrap());
                get_observed_ranges(&full_replay)
            })
            .reduce(reduce_ranges)
            .unwrap();

        let min_feature_variation = 0.25;
        for (feature_scope_id, feature_iterable) in [
            temporal_spatial_ranges.iter().enumerate(),
            nontemporal_spatial_ranges.iter().enumerate(),
            temporal_global_ranges.iter().enumerate(),
        ]
        .into_iter()
        .enumerate()
        {
            for (feature_id, &(min_val, max_val)) in feature_iterable {
                println!(
                    "{:?}",
                    ((feature_scope_id, feature_id), (min_val, max_val),)
                );
                if min_val != 0.0 {
                    assert!(min_val <= -min_feature_variation);
                }
                assert!(max_val >= min_feature_variation);
            }
        }
    }
}
