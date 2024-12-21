use crate::rules_engine::params::P;

#[derive(Debug, Clone)]
pub struct StepStats {
    pub terminal_points_scored: Option<[u32; P]>,
    pub normalized_terminal_points_scored: Option<[f32; P]>,
    pub energy_field_deltas: Vec<i32>,
    pub normalized_energy_field_deltas: Vec<f32>,
    pub nebula_energy_deltas: Vec<i32>,
    pub energy_void_field_deltas: Vec<i32>,

    pub units_lost_to_energy: u16,
    pub units_lost_to_collision: u16,

    pub noop_count: u16,
    pub move_count: u16,
    pub sap_count: u16,
    pub sap_direct_hits: u16,
    pub sap_adjacent_hits: u16,
}

#[derive(Debug, Clone)]
pub struct GameStats {
    /// Points scored at end of match
    pub terminal_points_scored: Vec<u32>,
    /// terminal_points_scored / point_tile_count / steps_in_match
    pub normalized_terminal_points_scored: Vec<f32>,
    /// Per-unit energy field deltas
    pub energy_field_deltas: Vec<i32>,
    /// energy_field_deltas normalized into the range [0, 1]
    pub normalized_energy_field_deltas: Vec<f32>,
    /// Per-unit nebula energy deltas
    pub nebula_energy_deltas: Vec<i32>,
    /// Per-unit energy void field deltas
    pub energy_void_field_deltas: Vec<i32>,

    pub units_lost_to_energy: u32,
    pub units_lost_to_collision: u32,

    pub noop_count: u32,
    pub move_count: u32,
    pub sap_count: u32,
    pub sap_direct_hits: u32,
    pub sap_adjacent_hits: u32,
}

impl GameStats {
    pub fn new() -> Self {
        Self {
            terminal_points_scored: Vec::new(),
            normalized_terminal_points_scored: Vec::new(),
            energy_field_deltas: Vec::new(),
            normalized_energy_field_deltas: Vec::new(),
            nebula_energy_deltas: Vec::new(),
            energy_void_field_deltas: Vec::new(),
            units_lost_to_energy: 0,
            units_lost_to_collision: 0,
            noop_count: 0,
            move_count: 0,
            sap_count: 0,
            sap_direct_hits: 0,
            sap_adjacent_hits: 0,
        }
    }

    pub fn extend(&mut self, stats: StepStats) {
        if let Some(terminal_points_scored) = stats.terminal_points_scored {
            self.terminal_points_scored
                .extend_from_slice(&terminal_points_scored);
            self.normalized_terminal_points_scored.extend_from_slice(
                &stats.normalized_terminal_points_scored.unwrap(),
            );
        }
        self.energy_field_deltas.extend(stats.energy_field_deltas);
        self.normalized_energy_field_deltas
            .extend(stats.normalized_energy_field_deltas);
        self.nebula_energy_deltas.extend(stats.nebula_energy_deltas);
        self.energy_void_field_deltas
            .extend(stats.energy_void_field_deltas);

        self.units_lost_to_energy += u32::from(stats.units_lost_to_energy);
        self.units_lost_to_collision +=
            u32::from(stats.units_lost_to_collision);

        self.noop_count += u32::from(stats.noop_count);
        self.move_count += u32::from(stats.move_count);
        self.sap_count += u32::from(stats.sap_count);
        self.sap_direct_hits += u32::from(stats.sap_direct_hits);
        self.sap_adjacent_hits += u32::from(stats.sap_adjacent_hits);
    }

    fn total_actions(&self) -> u32 {
        self.noop_count + self.move_count + self.sap_count
    }

    pub fn noop_frequency(&self) -> f32 {
        self.noop_count as f32 / self.total_actions() as f32
    }

    pub fn move_frequency(&self) -> f32 {
        self.move_count as f32 / self.total_actions() as f32
    }

    pub fn sap_frequency(&self) -> f32 {
        self.sap_count as f32 / self.total_actions() as f32
    }

    pub fn sap_direct_hits_frequency(&self) -> f32 {
        if self.sap_count > 0 {
            self.sap_direct_hits as f32 / self.sap_count as f32
        } else {
            0.0
        }
    }

    pub fn sap_adjacent_hits_frequency(&self) -> f32 {
        if self.sap_count > 0 {
            self.sap_adjacent_hits as f32 / self.sap_count as f32
        } else {
            0.0
        }
    }
}
