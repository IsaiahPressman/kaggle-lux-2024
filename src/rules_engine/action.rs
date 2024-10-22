pub enum Action {
    NoOp,
    Up,
    Right,
    Down,
    Left,
    Sap([isize; 2]),
}
