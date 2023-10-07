use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

// The name of the module must be the same as the rust package name
#[pymodule]
fn rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    Ok(())
}