use crate::rules_engine::state::{Observation, Unit};
use itertools::Itertools;
use numpy::ndarray::{
    ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, Axis,
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
    RelicNodePoint,
}

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum GlobalFeature {
    MyTeamPoints,
    OppTeamPoints,
    MyTeamWins,
    OppTeamWins,
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
) {
    for ((obs, team_spatial_out), team_global_out) in observations
        .iter()
        .zip_eq(spatial_out.axis_iter_mut(Axis(0)))
        .zip_eq(global_out.axis_iter_mut(Axis(0)))
    {
        write_team_obs(team_spatial_out, team_global_out, obs);
    }
}

fn write_team_obs(
    mut spatial_out: ArrayViewMut3<f32>,
    mut global_out: ArrayViewMut1<f32>,
    obs: &Observation,
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
                obs.asteroids
                    .iter()
                    .for_each(|a| slice[a.as_index()] += 1.0);
            },
            Nebula => {
                obs.nebulae.iter().for_each(|n| slice[n.as_index()] += 1.0);
            },
            // TODO: Pair program this
            RelicNode => {
                todo!()
            },
            RelicNodePoint => {
                todo!()
            },
        }
    }

    for (gf, value) in GlobalFeature::iter().zip_eq(global_out.iter_mut()) {
        match gf {
            MyTeamPoints => {
                *value = obs.team_points[obs.team_id] as f32;
            },
            OppTeamPoints => {
                *value = obs.team_points[1 - obs.team_id] as f32;
            },
            MyTeamWins => {
                *value = obs.team_wins[obs.team_id] as f32;
            },
            OppTeamWins => {
                *value = obs.team_wins[1 - obs.team_id] as f32;
            },
        }
    }
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
