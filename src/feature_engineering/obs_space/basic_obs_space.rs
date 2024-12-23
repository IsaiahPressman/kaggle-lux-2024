use crate::feature_engineering::memory::Memory;
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
    MyUnitEnergy,
    OppUnitEnergy,
    MyUnitEnergyMin,
    OppUnitEnergyMin,
    MyUnitEnergyMax,
    OppUnitEnergyMax,
    Asteroid,
    Nebula,
    RelicNode,
    RelicNodeExplored,
    RelicNodePoints,      // Guesstimate of the points value
    RelicNodePointsKnown, // Whether the guessed value is known to be correct
    EnergyField,
}

#[derive(Debug, Clone, Copy, EnumIter)]
enum GlobalFeature {
    MyTeamPoints = 0,
    OppTeamPoints = 1,
    MyTeamId = 2,
    // TODO: Discretize team_wins features
    MyTeamWins = 4,
    OppTeamWins = 5,
    NebulaTileVisionReduction = 6,
    NebulaTileEnergyReduction = 10,
    UnitSapDropoffFactor = 13,
    End = 16,
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
) {
    for (((obs, mem), team_spatial_out), team_global_out) in observations
        .iter()
        .zip_eq(memories)
        .zip_eq(spatial_out.outer_iter_mut())
        .zip_eq(global_out.outer_iter_mut())
    {
        write_team_obs(team_spatial_out, team_global_out, obs, mem);
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
            Asteroid => {
                // TODO use memory
                obs.asteroids.iter().for_each(|a| slice[a.as_index()] = 1.0);
            },
            Nebula => {
                // TODO use memory
                obs.nebulae.iter().for_each(|n| slice[n.as_index()] = 1.0);
            },
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
            MyTeamPoints => {
                global_result[gf as usize] =
                    obs.team_points[obs.team_id] as f32;
            },
            OppTeamPoints => {
                global_result[gf as usize] =
                    obs.team_points[1 - obs.team_id] as f32;
            },
            MyTeamId => {
                let onehot_team_id = match obs.team_id {
                    0 => [1., 0.],
                    1 => [0., 1.],
                    _ => unreachable!(),
                };
                global_result[gf as usize..next_gf as usize]
                    .copy_from_slice(&onehot_team_id);
            },
            MyTeamWins => {
                global_result[gf as usize] = obs.team_wins[obs.team_id] as f32;
            },
            OppTeamWins => {
                global_result[gf as usize] =
                    obs.team_wins[1 - obs.team_id] as f32;
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
            UnitSapDropoffFactor => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.get_unit_sap_dropoff_factor_weights(),
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
                End => panic!("End should be the last feature"),
                _ => {
                    assert_eq!(feature as isize, next_feature as isize - 1)
                },
            }
        }
    }
}
