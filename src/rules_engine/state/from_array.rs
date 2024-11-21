use super::state_data::Pos;
use crate::rules_engine::state::EnergyNode;
use itertools::Itertools;
use numpy::ndarray::{ArrayView1, ArrayView2};

const NOT_VISIBLE: i32 = -1;
const EMPTY_TILE: i32 = 0;
const NEBULA_TILE: i32 = 1;
const ASTEROID_TILE: i32 = 2;

pub fn get_asteroids(tile_type: ArrayView2<i32>) -> Vec<Pos> {
    tile_type
        .indexed_iter()
        .filter_map(|((x, y), &tile)| {
            filter_map_tile(x, y, tile, ASTEROID_TILE)
        })
        .collect()
}

pub fn get_nebulae(tile_type: ArrayView2<i32>) -> Vec<Pos> {
    tile_type
        .indexed_iter()
        .filter_map(|((x, y), &tile)| filter_map_tile(x, y, tile, NEBULA_TILE))
        .collect()
}

fn filter_map_tile(
    x: usize,
    y: usize,
    tile_type: i32,
    target: i32,
) -> Option<Pos> {
    if tile_type == target {
        Some(Pos::new(x, y))
    } else if tile_type == NOT_VISIBLE
        || tile_type == EMPTY_TILE
        || tile_type == ASTEROID_TILE
        || tile_type == NEBULA_TILE
    {
        None
    } else {
        panic!("Unrecognized tile type: {}", tile_type)
    }
}

pub fn get_energy_nodes(
    locations: ArrayView2<i16>,
    node_fns: ArrayView2<f32>,
    mask: ArrayView1<bool>,
) -> Vec<EnergyNode> {
    locations
        .outer_iter()
        .zip_eq(node_fns.outer_iter())
        .zip_eq(mask.iter())
        .filter(|&(_, &mask)| mask)
        .map(|((loc, node_fn), _)| {
            match (loc.as_slice().unwrap(), node_fn.as_slice().unwrap()) {
                (&[x, y], &[f_id, a, b, c]) => EnergyNode::new(
                    Pos::new(x as usize, y as usize),
                    f_id as u8,
                    [a, b, c],
                ),
                (loc, node_fn) => {
                    panic!("Unexpected (loc, node_fn): {:?}", (loc, node_fn))
                },
            }
        })
        .collect()
}
