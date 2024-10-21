use super::action::Action;
use super::params::Params;
use super::state::{Pos, State};
use itertools::zip_eq;
use numpy::ndarray::Array2;

pub fn step(
    state: &mut State,
    actions: &[Vec<Action>; 2],
    params: &Params,
    energy_node_deltas: Option<Vec<[usize; 2]>>,
) {
    // TODO: Do we need to precompute energy features
    if state.match_steps == 0 {
        state.units = [Vec::new(), Vec::new()];
    }
    remove_dead_units(state);
    move_units(
        state,
        actions,
        params.unit_move_cost,
        state.get_asteroid_mask(params.map_height, params.map_width),
    );
    unimplemented!("sap actions");
}

fn remove_dead_units(state: &mut State) {
    state.units[0].retain(|u| u.energy >= 0);
    state.units[1].retain(|u| u.energy >= 0);
}

fn move_units(
    state: &mut State,
    actions: &[Vec<Action>; 2],
    unit_move_cost: i32,
    asteroid_mask: Array2<bool>,
) {
    for p in [0, 1] {
        for (u, a) in zip_eq(state.units[p].iter_mut(), actions[p].iter()) {
            if u.energy <= unit_move_cost {
                continue;
            }

            let new_pos = match a {
                Action::Up => Pos {
                    x: u.pos.x,
                    y: u.pos.y.saturating_sub(1),
                },
                Action::Right => Pos {
                    x: u.pos.x + 1,
                    y: u.pos.y,
                },
                Action::Down => Pos {
                    x: u.pos.x,
                    y: u.pos.y + 1,
                },
                Action::Left => Pos {
                    x: u.pos.x.saturating_sub(1),
                    y: u.pos.y,
                },
                Action::NoOp | Action::Sap(_) => continue,
            };
            if asteroid_mask[new_pos.as_index()] {
                continue;
            }

            u.pos = new_pos;
            u.energy -= unit_move_cost;
        }
    }
}
