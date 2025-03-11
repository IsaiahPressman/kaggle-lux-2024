use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use strum::{EnumIter, IntoEnumIterator};

#[pyclass(eq, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum SapMasking {
    PointTiles,
    OppUnitFrontier,
}

#[pymethods]
impl SapMasking {
    fn __str__(&self) -> PyResult<String> {
        let (_, name) = self.__pyo3__repr__().split_once(".").unwrap();
        Ok(name.to_string())
    }

    #[staticmethod]
    fn list() -> Vec<Self> {
        Self::iter().collect()
    }

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        for mode in SapMasking::iter() {
            if mode.__str__()? == s {
                return Ok(mode);
            }
        }
        Err(PyValueError::new_err(format!("Invalid SapMasking '{s}'")))
    }
}
