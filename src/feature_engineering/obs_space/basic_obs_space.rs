use crate::feature_engineering::memory::Memory;
use crate::rules_engine::param_ranges::{
    PARAM_RANGES, UNIT_SAP_COST_MAX, UNIT_SAP_COST_MIN,
};
use crate::rules_engine::params::{KnownVariableParams, FIXED_PARAMS};
use crate::rules_engine::state::{Observation, Unit};
use itertools::Itertools;
use numpy::ndarray::{
    ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, Zip,
};
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum SpatialFeature {
    Visible,
    MyUnitCount,
    OppUnitCount,
    // Dead units still provide vision
    MyDeadUnitCount,
    OppDeadUnitCount,
    MyUnitEnergy,
    OppUnitEnergy,
    MyUnitEnergyMin,
    OppUnitEnergyMin,
    MyUnitEnergyMax,
    OppUnitEnergyMax,
    Asteroid,
    Nebula,
    TileExplored,
    RelicNode,
    RelicNodeExplored,
    RelicNodePoints,      // Guesstimate of the points value
    RelicNodePointsKnown, // Whether the guessed value is known to be correct
    EnergyField,
}

#[derive(Debug, Clone, Copy, EnumIter)]
enum GlobalFeature {
    // Visible features
    MyTeamId = 0,
    MyTeamPoints = 2,
    OppTeamPoints = 3,
    MyTeamWins = 4,
    OppTeamWins = 7,
    MatchSteps = 10,
    // Known parameters
    UnitMoveCost = 11,
    UnitSapCost = 16,
    UnitSapRange = 17,
    UnitSensorRange = 22,
    // Estimated / inferred features
    UnitSapDropoffFactor = 25,
    NebulaTileVisionReduction = 28,
    NebulaTileEnergyReduction = 32,
    NebulaTileDriftSpeed = 35,
    EnergyNodeDriftSpeed = 39,
    End = 44,
}

// Normalizing constants
const UNIT_COUNT_NORM: f32 = 4.0;
const UNIT_ENERGY_NORM: f32 = 400.0;
const UNIT_ENERGY_MIN_BASELINE: f32 = 0.1;
const ENERGY_FIELD_NORM: f32 = 7.0;

/// Writes into spatial_out of shape (teams, s_channels, map_width, map_height) and
/// global_out of shape (teams, g_channels)
pub fn write_obs_arrays(
    mut spatial_out: ArrayViewMut4<f32>,
    mut global_out: ArrayViewMut2<f32>,
    observations: &[Observation],
    memories: &[Memory],
    params: &KnownVariableParams,
) {
    for (((obs, mem), team_spatial_out), team_global_out) in observations
        .iter()
        .zip_eq(memories)
        .zip_eq(spatial_out.outer_iter_mut())
        .zip_eq(global_out.outer_iter_mut())
    {
        write_team_obs(team_spatial_out, team_global_out, obs, mem, params);
    }
}

pub fn get_spatial_feature_count() -> usize {
    SpatialFeature::COUNT
}

pub fn get_global_feature_count() -> usize {
    GlobalFeature::End as usize
}

fn write_team_obs(
    mut spatial_out: ArrayViewMut3<f32>,
    mut global_out: ArrayViewMut1<f32>,
    obs: &Observation,
    mem: &Memory,
    params: &KnownVariableParams,
) {
    use GlobalFeature::*;
    use SpatialFeature::*;

    for (sf, mut slice) in
        SpatialFeature::iter().zip_eq(spatial_out.outer_iter_mut())
    {
        match sf {
            Visible => {
                slice.assign(
                    &obs.sensor_mask.map(|v| if *v { 1.0 } else { 0.0 }),
                );
            },
            MyUnitCount => {
                write_unit_counts(slice, obs.get_my_units());
            },
            OppUnitCount => {
                write_unit_counts(slice, obs.get_opp_units());
            },
            MyDeadUnitCount => {
                write_dead_unit_counts(slice, obs.get_my_units());
            },
            OppDeadUnitCount => {
                write_dead_unit_counts(slice, obs.get_opp_units());
            },
            MyUnitEnergy => {
                write_unit_energies(slice, obs.get_my_units());
            },
            OppUnitEnergy => {
                write_unit_energies(slice, obs.get_opp_units());
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
            RelicNodePoints => slice.assign(mem.get_relic_points_map()),
            RelicNodePointsKnown => Zip::from(&mut slice)
                .and(mem.get_known_relic_points_map())
                .for_each(|out, &explored| {
                    *out = if explored { 1.0 } else { 0.0 }
                }),
            EnergyField => Zip::from(&mut slice)
                .and(mem.get_energy_field())
                .for_each(|out, &energy| {
                    if let Some(e) = energy {
                        *out = e as f32 / ENERGY_FIELD_NORM
                    }
                }),
        }
    }

    let mut global_result: Vec<f32> = vec![0.0; End as usize];
    for (gf, next_gf) in GlobalFeature::iter().tuple_windows() {
        match gf {
            MyTeamId => {
                let onehot_team_id = match obs.team_id {
                    0 => [1., 0.],
                    1 => [0., 1.],
                    _ => unreachable!(),
                };
                global_result[gf as usize..next_gf as usize]
                    .copy_from_slice(&onehot_team_id);
            },
            MyTeamPoints => {
                global_result[gf as usize] =
                    obs.team_points[obs.team_id] as f32;
            },
            OppTeamPoints => {
                global_result[gf as usize] =
                    obs.team_points[1 - obs.team_id] as f32;
            },
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
    global_out
        .iter_mut()
        .zip_eq(global_result)
        .for_each(|(out, v)| *out = v);
}

fn write_unit_counts(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units
        .iter()
        .filter(|u| u.alive())
        .for_each(|u| slice[u.pos.as_index()] += 1. / UNIT_COUNT_NORM);
}

fn write_dead_unit_counts(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units
        .iter()
        .filter(|u| !u.alive())
        .for_each(|u| slice[u.pos.as_index()] += 1. / UNIT_COUNT_NORM);
}

fn write_unit_energies(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units.iter().filter(|u| u.alive()).for_each(|u| {
        slice[u.pos.as_index()] += u.energy as f32 / UNIT_ENERGY_NORM
    });
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
    use crate::rules_engine::param_ranges::PARAM_RANGES;
    use crate::rules_engine::params::P;
    use GlobalFeature::*;

    #[test]
    fn test_global_feature_indices() {
        for (feature, next_feature) in GlobalFeature::iter().tuple_windows() {
            match feature {
                MyTeamId => {
                    assert_eq!(
                        next_feature as isize - feature as isize,
                        P as isize
                    );
                },
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
