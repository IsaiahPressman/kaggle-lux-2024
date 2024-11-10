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
    pub fn as_index(self) -> usize {
        match self {
            NoOp => 0,
            Up => 1,
            Right => 2,
            Down => 3,
            Left => 4,
            Sap(_) => 5,
        }
    }

    /// Get the action's move deltas [dx, dy], panicking on NoOp or Sap
    pub fn as_move_delta(self) -> [isize; 2] {
        match self {
            Up => [0, -1],
            Right => [1, 0],
            Down => [0, 1],
            Left => [-1, 0],
            NoOp | Sap(_) => {
                panic!("invalid move action: {:?}", self);
            },
        }
    }
}

impl From<[isize; 3]> for Action {
    fn from(value: [isize; 3]) -> Self {
        match value {
            [0, 0, 0] => NoOp,
            [1, 0, 0] => Up,
            [2, 0, 0] => Right,
            [3, 0, 0] => Down,
            [4, 0, 0] => Left,
            [5, x, y] => Sap([x, y]),
            a => panic!("Invalid action: {:?}", a),
        }
    }
}
