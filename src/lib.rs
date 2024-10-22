#![allow(dead_code)] // TODO: Re-enable this lint later
mod rules_engine;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// Prints a message.
#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("Hello from rux-2024!".into())
}

/// Makes an array.
#[pyfunction]
fn hello_numpy_world(py: Python<'_>) -> PyResult<Bound<'_, PyArray2<f32>>> {
    let mut arr = Array2::<f32>::zeros((4, 2));
    arr[[0, 0]] = 1.;
    arr[[3, 1]] = 2.;
    Ok(arr.into_pyarray_bound(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    m.add_function(wrap_pyfunction!(hello_numpy_world, m)?)?;
    Ok(())
}
