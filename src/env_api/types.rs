use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyArray5};
use pyo3::Bound;
use std::collections::HashMap;

pub type PyStatsOutputs<'py> = (
    HashMap<String, f32>,
    HashMap<String, Bound<'py, PyArray1<f32>>>,
);
pub type PyEnvOutputs<'py> = (
    (Bound<'py, PyArray5<f32>>, Bound<'py, PyArray3<f32>>),
    (
        Bound<'py, PyArray4<bool>>,
        Bound<'py, PyArray5<bool>>,
        Bound<'py, PyArray4<isize>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<bool>>,
    ),
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<bool>>,
    Option<PyStatsOutputs<'py>>,
);
