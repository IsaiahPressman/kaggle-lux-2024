use crate::rules_engine::env::TerminationMode;
use crate::rules_engine::params::P;
use crate::rules_engine::state::GameResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use strum::{EnumIter, IntoEnumIterator};
use RewardSpace::{FinalWinner, MatchWinner, PointsScored};

#[pyclass(eq, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum RewardSpace {
    FinalWinner,
    MatchWinner,
    PointsScored,
}

impl RewardSpace {
    pub fn termination_mode(&self) -> TerminationMode {
        match self {
            FinalWinner => TerminationMode::ThirdMatchWin,
            MatchWinner | PointsScored => TerminationMode::FinalStep,
        }
    }

    pub fn get_reward(&self, result: GameResult) -> [f32; P] {
        match self {
            FinalWinner => Self::from_winner(result.final_winner),
            MatchWinner => Self::from_winner(result.match_winner),
            PointsScored => result.points_scored.map(|scored| scored as f32),
        }
    }

    fn from_winner(winner: Option<u8>) -> [f32; P] {
        if let Some(p) = winner {
            match p {
                0 => [1.0, -1.0],
                1 => [-1.0, 1.0],
                p => panic!("Unexpected winner {p}"),
            }
        } else {
            [0.0, 0.0]
        }
    }
}

#[pymethods]
impl RewardSpace {
    #[staticmethod]
    fn list() -> Vec<Self> {
        Self::iter().collect()
    }

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        for rs in RewardSpace::iter() {
            let (_, name) = rs.__pyo3__repr__().split_once(".").unwrap();
            if name == s {
                return Ok(rs);
            }
        }
        Err(PyValueError::new_err(format!("Invalid RewardSpace '{s}'")))
    }
}
