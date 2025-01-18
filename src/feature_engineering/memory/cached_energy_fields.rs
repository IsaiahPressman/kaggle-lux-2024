use crate::rules_engine::env::get_energy_field;
use crate::rules_engine::params::FIXED_PARAMS;
use crate::rules_engine::state::{EnergyNode, Pos};
use itertools::Itertools;
use numpy::ndarray::Array2;
use std::sync::LazyLock;

pub static CACHED_ENERGY_FIELDS: LazyLock<Vec<(Pos, Array2<i32>)>> =
    LazyLock::new(generate_cached_energy_fields);
const MAP_SIZE: usize = {
    assert!(
        FIXED_PARAMS.map_width == FIXED_PARAMS.map_height,
        "Map isn't square"
    );
    FIXED_PARAMS.map_width
};
const FIXED_ENERGY_NODE_FNS: [(u8, f32, f32, f32);
    FIXED_PARAMS.max_energy_nodes / 2] =
    [(0, 1.2, 1., 4.), (0, 0., 0., 0.), (0, 0., 0., 0.)];

fn generate_cached_energy_fields() -> Vec<(Pos, Array2<i32>)> {
    let mut result = Vec::new();
    for x in 0..MAP_SIZE {
        for y in 0..MAP_SIZE - x {
            let pos = Pos::new(x, y);
            let node_positions = [
                pos,
                Pos::default(),
                Pos::default(),
                pos.reflect(FIXED_PARAMS.map_size),
                Pos::default(),
                Pos::default(),
            ];
            let energy_nodes = FIXED_ENERGY_NODE_FNS
                .into_iter()
                .chain(FIXED_ENERGY_NODE_FNS.into_iter())
                .zip_eq(node_positions.into_iter())
                .map(|(node_fn, node_pos)| {
                    EnergyNode::from_pos_and_energy_fn(node_pos, node_fn)
                })
                .collect_vec();
            let energy_field = get_energy_field(&energy_nodes, &FIXED_PARAMS);
            result.push((pos, energy_field));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_engineering::replay::load_replay;
    use rstest::rstest;
    use std::path::PathBuf;

    #[test]
    fn test_cached_energy_fields_len() {
        assert_eq!(CACHED_ENERGY_FIELDS.len(), MAP_SIZE * (MAP_SIZE + 1) / 2);
    }

    #[test]
    fn test_cached_energy_fields_complete() {
        for pos in (0..FIXED_PARAMS.map_width)
            .cartesian_product(0..FIXED_PARAMS.map_height)
            .map(Pos::from)
        {
            assert!(CACHED_ENERGY_FIELDS.iter().any(|(cache_pos, _)| {
                pos == *cache_pos
                    || pos.reflect(FIXED_PARAMS.map_size) == *cache_pos
            }));
        }
    }

    #[test]
    fn test_cached_energy_fields_unique() {
        for (pos, energy_field) in CACHED_ENERGY_FIELDS.iter() {
            for (_, other_field) in CACHED_ENERGY_FIELDS
                .iter()
                .filter(|(other_pos, _)| other_pos != pos)
            {
                assert_ne!(energy_field, other_field);
            }
        }
    }

    #[rstest]
    #[ignore = "slow"]
    fn test_cached_energy_fields_values(
        #[files("src/feature_engineering/test_data/*.json")] path: PathBuf,
    ) {
        let full_replay = load_replay(path);
        for field in full_replay.get_energy_fields() {
            assert!(CACHED_ENERGY_FIELDS
                .iter()
                .any(|(_, cached_field)| field == cached_field));
        }
    }
}
