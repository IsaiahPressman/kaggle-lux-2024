use strum_macros::{EnumCount, EnumIter};
use Action::{Down, Left, NoOp, Right, Sap, Up};

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCount, EnumIter)]
pub enum Action {
    NoOp,
    Up,
    Right,
    Down,
    Left,
    Sap([isize; 2]),
}

impl Action {
    /// Get the action's move deltas [dx, dy], returning None on NoOp or Sap
    pub fn as_move_delta(self) -> Option<[isize; 2]> {
        match self {
            Up => Some([0, -1]),
            Right => Some([1, 0]),
            Down => Some([0, 1]),
            Left => Some([-1, 0]),
            NoOp | Sap(_) => None,
        }
    }
}

impl From<[isize; 3]> for Action {
    fn from(value: [isize; 3]) -> Self {
        match value {
            [0, _, _] => NoOp,
            [1, _, _] => Up,
            [2, _, _] => Right,
            [3, _, _] => Down,
            [4, _, _] => Left,
            [5, x, y] => Sap([x, y]),
            a => panic!("Invalid action: {a:?}"),
        }
    }
}
