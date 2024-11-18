use crate::feature_engineering::memory::Memory;
use crate::rules_engine::state::{Observation, Unit};
use itertools::Itertools;
use numpy::ndarray::{
    ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, Axis, Zip,
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
    Asteroid,
    Nebula,
    RelicNode,
    RelicNodeExplored,
    RelicNodePoint,      // Guesstimate of the value
    KnownRelicNodePoint, // Known Relic Node points
    EnergyField,
}

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum GlobalFeature {
    MyTeamPoints = 0,
    OppTeamPoints = 1,
    MyTeamWins = 2,
    OppTeamWins = 3,
    NebulaTileVisionReduction = 4,
    NebulaTileEnergyReduction = 8,
    UnitSapDropoffFactor = 11,
    End = 14,
}

// Normalizing constants
const UNIT_COUNT_NORM: f32 = 4.0;
const UNIT_ENERGY_NORM: f32 = 400.0;

/// Writes into spatial_out of shape (teams, s_channels, map_width, map_height) and
/// global_out of shape (teams, g_channels)
pub fn write_basic_obs_space(
    spatial_out: &mut ArrayViewMut4<f32>,
    global_out: &mut ArrayViewMut2<f32>,
    observations: &[Observation; 2],
    memories: &[Memory; 2],
) {
    for (((obs, mem), team_spatial_out), team_global_out) in observations
        .iter()
        .zip_eq(memories)
        .zip_eq(spatial_out.axis_iter_mut(Axis(0)))
        .zip_eq(global_out.axis_iter_mut(Axis(0)))
    {
        write_team_obs(team_spatial_out, team_global_out, obs, mem);
    }
}

fn write_team_obs(
    mut spatial_out: ArrayViewMut3<f32>,
    mut global_out: ArrayViewMut1<f32>,
    obs: &Observation,
    mem: &Memory,
) {
    use GlobalFeature::*;
    use SpatialFeature::*;

    let opp = 1 - obs.team_id;
    for (sf, mut slice) in
        SpatialFeature::iter().zip_eq(spatial_out.axis_iter_mut(Axis(0)))
    {
        match sf {
            Visible => {
                slice.assign(
                    &obs.sensor_mask.map(|v| if *v { 1.0 } else { 0.0 }),
                );
            },
            MyUnitCount => {
                write_unit_counts(slice, &obs.units[obs.team_id]);
            },
            OppUnitCount => {
                write_unit_counts(slice, &obs.units[opp]);
            },
            MyUnitEnergy => {
                write_unit_energies(slice, &obs.units[obs.team_id]);
            },
            OppUnitEnergy => {
                write_unit_energies(slice, &obs.units[opp]);
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
                mem.relic_nodes
                    .relic_nodes
                    .iter()
                    .for_each(|r| slice[r.as_index()] = 1.0);
            },
            RelicNodeExplored => Zip::from(&mut slice)
                .and(&mem.relic_nodes.explored_nodes_map)
                .for_each(|out, &explored| {
                    *out = if explored { 1.0 } else { 0.0 }
                }),
            RelicNodePoint => slice.assign(&mem.relic_nodes.points_map),
            KnownRelicNodePoint => Zip::from(&mut slice)
                .and(&mem.relic_nodes.known_points_map)
                .for_each(|out, &explored| {
                    *out = if explored { 1.0 } else { 0.0 }
                }),
            EnergyField => Zip::from(&mut slice)
                .and(&mem.energy_field.energy_field)
                .for_each(|out, &energy| {
                    if let Some(e) = energy {
                        *out = e as f32
                    }
                }),
        }
    }

    let mut global_result: Vec<f32> = Vec::with_capacity(End as usize);
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
            MyTeamWins => {
                global_result[gf as usize] = obs.team_wins[obs.team_id] as f32;
            },
            OppTeamWins => {
                global_result[gf as usize] =
                    obs.team_wins[1 - obs.team_id] as f32;
            },
            NebulaTileVisionReduction => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.hidden_parameters
                        .nebula_tile_vision_reduction
                        .get_weighted_possibilities(),
                );
            },
            NebulaTileEnergyReduction => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.hidden_parameters
                        .nebula_tile_energy_reduction
                        .get_weighted_possibilities(),
                );
            },
            UnitSapDropoffFactor => {
                global_result[gf as usize..next_gf as usize].copy_from_slice(
                    &mem.hidden_parameters
                        .unit_sap_dropoff_factor
                        .get_weighted_possibilities(),
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
        .for_each(|u| slice[u.pos.as_index()] += 1. / UNIT_COUNT_NORM);
}

fn write_unit_energies(mut slice: ArrayViewMut2<f32>, units: &[Unit]) {
    units.iter().for_each(|u| {
        slice[u.pos.as_index()] += u.energy as f32 / UNIT_ENERGY_NORM
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules_engine::param_ranges::PARAM_RANGES;
    use GlobalFeature::*;

    #[test]
    fn test_global_feature_indices() {
        for (feature, next_feature) in GlobalFeature::iter().tuple_windows() {
            match feature {
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
                _ => {
                    assert_eq!(feature as isize, next_feature as isize - 1)
                },
            }
        }
    }
}
