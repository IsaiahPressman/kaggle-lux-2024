#[derive(Debug, Clone)]
pub struct StepStats {
    /// Points scored at end of match
    pub terminal_points_scored: Option<[u32; 2]>,
    /// Per-unit energy field deltas
    pub energy_field_deltas: Vec<i32>,
    /// Per-unit nebula energy deltas
    pub nebula_energy_deltas: Vec<i32>,
    /// Per-unit energy void field deltas
    pub energy_void_field_deltas: Vec<i32>,

    pub units_lost_to_energy: u16,
    pub units_lost_to_collision: u16,

    pub noop_count: u16,
    pub move_count: u16,
    pub sap_count: u16,
    pub sap_direct_hits: u16,
    pub sap_adjacent_hits: u16,
}

impl StepStats {
    fn total_actions(&self) -> u16 {
        self.noop_count + self.move_count + self.sap_count
    }
}
