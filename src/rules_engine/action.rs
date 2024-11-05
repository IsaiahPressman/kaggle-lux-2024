use Action::{Down, Left, NoOp, Right, Sap, Up};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    NoOp,
    Up,
    Right,
    Down,
    Left,
    Sap([isize; 2]),
}

impl From<[isize; 3]> for Action {
    fn from(value: [isize; 3]) -> Self {
        match value {
            [0, ..] => NoOp,
            [1, ..] => Up,
            [2, ..] => Right,
            [3, ..] => Down,
            [4, ..] => Left,
            [5, x, y] => Sap([x, y]),
            [a, ..] => panic!("Invalid action type: {}", a),
        }
    }
}
