use crate::rules_engine::params::{MAP_HEIGHT, MAP_WIDTH, TEAMS};
use crate::rules_engine::state::{Observation, Unit};
use numpy::ndarray::{Array4, ArrayViewMut2, ArrayViewMut3, Axis};
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

#[derive(Debug, Clone, Copy, EnumCount, EnumIter)]
enum Feature {
    Visible,
    MyUnitCount,
    OppUnitCount,
    MyUnitEnergy,
    OppUnitEnergy,
    Asteroid,
    Nebula,
    RelicNode,
    RelicNodePoint,
    MyTeamPoints,
    OppTeamPoints,
    MyTeamWins,
    OppTeamWins,
}

// Normalizing constants
const UNIT_COUNT_NORM: f32 = 4.0;
const UNIT_ENERGY_NORM: f32 = 400.0;

/// Returns an array of shape (teams, channels, 24, 24)
pub fn basic_obs_space(observations: [Observation; 2]) -> Array4<f32> {
    let mut out = Array4::zeros((TEAMS, Feature::COUNT, MAP_WIDTH, MAP_HEIGHT));
    for (team, obs) in observations.iter().enumerate() {
        write_team_obs(&mut out.index_axis_mut(Axis(0), team), team, obs);
    }
    out
}

fn write_team_obs(
    out: &mut ArrayViewMut3<f32>,
    team: usize,
    obs: &Observation,
) {
    use Feature::*;

    let opp = 1 - team;
    for f in Feature::iter() {
        let mut slice = out.index_axis_mut(Axis(0), f as usize);
        match f {
            Visible => {
                slice.assign(
                    &obs.sensor_mask.map(|v| if *v { 1.0 } else { 0.0 }),
                );
            },
            MyUnitCount => {
                write_unit_counts(&mut slice, &obs.units[team]);
            },
            OppUnitCount => {
                write_unit_counts(&mut slice, &obs.units[opp]);
            },
            MyUnitEnergy => {
                write_unit_energies(&mut slice, &obs.units[team]);
            },
            OppUnitEnergy => {
                write_unit_energies(&mut slice, &obs.units[opp]);
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
            MyTeamPoints => {
                todo!()
            },
            OppTeamPoints => {
                todo!()
            },
            MyTeamWins => {
                todo!()
            },
            OppTeamWins => {
                todo!()
            },
        }
    }
}

fn write_unit_counts(slice: &mut ArrayViewMut2<f32>, units: &[Unit]) {
    units
        .iter()
        .for_each(|u| slice[u.pos.as_index()] += 1. / UNIT_COUNT_NORM);
}

fn write_unit_energies(slice: &mut ArrayViewMut2<f32>, units: &[Unit]) {
    units.iter().for_each(|u| {
        slice[u.pos.as_index()] += u.energy as f32 / UNIT_ENERGY_NORM
    });
}
