mod env_api;
mod feature_engineering;
pub mod izip_eq;
mod rules_engine;

use crate::env_api::FeatureEngineeringEnv;
use crate::feature_engineering::obs_space::basic_obs_space::{
    get_global_feature_count, get_spatial_feature_count,
};
use crate::feature_engineering::reward_space::RewardSpace;
use env_api::ParallelEnv;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// Prints a message
#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("Hello from rux-ai-s3!".into())
}

/// Makes an array
#[pyfunction]
fn hello_numpy_world(py: Python<'_>) -> PyResult<Bound<'_, PyArray2<f32>>> {
    let mut arr = Array2::<f32>::zeros((4, 2));
    arr[[0, 0]] = 1.;
    arr[[3, 1]] = 2.;
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction(name = "get_spatial_feature_count")]
fn get_spatial_feature_count_py() -> usize {
    get_spatial_feature_count()
}

#[pyfunction(name = "get_global_feature_count")]
fn get_global_feature_count_py() -> usize {
    get_global_feature_count()
}

/// A Python module implemented in Rust
#[pymodule]
fn lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    m.add_function(wrap_pyfunction!(hello_numpy_world, m)?)?;

    m.add_class::<RewardSpace>()?;
    m.add_class::<ParallelEnv>()?;
    m.add_class::<FeatureEngineeringEnv>()?;
    m.add_function(wrap_pyfunction!(get_spatial_feature_count_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_global_feature_count_py, m)?)?;
    Ok(())
}
