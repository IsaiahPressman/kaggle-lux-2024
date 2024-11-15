use crate::rules_engine::state::{Observation, Pos};
use itertools::Itertools;
use numpy::ndarray::{Array2, Zip};

/// Tracks everything known by a player currently about the energy field
#[derive(Debug)]
pub struct EnergyFieldMemory {
    pub energy_field: Array2<Option<i32>>,
    map_size: [usize; 2],
}

impl EnergyFieldMemory {
    pub fn new(map_size: [usize; 2]) -> Self {
        EnergyFieldMemory {
            energy_field: Array2::default(map_size),
            map_size,
        }
    }

    pub fn update_memory(&mut self, obs: &Observation) {
        let mut new_energy_field = self.energy_field.clone();
        Zip::from(&mut new_energy_field)
            .and(&obs.energy_field)
            .for_each(|new_energy_field, &energy| {
                if energy.is_some() {
                    *new_energy_field = energy;
                }
            });

        symmetrize(&mut new_energy_field);
        // TODO: Check if new == old, ignored None values
        //  If so, carry on. Otherwise, reset new using just new obs data and
        //  redo symmetrization
    }
}

fn symmetrize(energy_field: &mut Array2<Option<i32>>) {
    let map_size = [energy_field.nrows(), energy_field.ncols()];
    for pos in (0..energy_field.nrows())
        .cartesian_product(0..energy_field.ncols())
        .map(|(x, y)| Pos::new(x, y))
    {
        if let Some(e) = energy_field[pos.as_index()] {
            energy_field[pos.reflect(map_size).as_index()].get_or_insert(e);
        }
    }
}
