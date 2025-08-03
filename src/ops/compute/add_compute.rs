use ndarray::ScalarOperand;
use std::fmt::Debug;
use std::ops::{Add, Deref};

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn add_compute_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T>,
{
    let lhs_raw = lhs_tensor.get_raw_data();
    let rhs_raw = rhs_tensor.get_raw_data();

    let new_raw = lhs_raw.deref() + rhs_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn add_compute_tensor_scalar<T, S>(tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    T: DTComp + Clone + Add<S, Output = T> + ScalarOperand + Debug,
    S: ScalarOperand,
{
    let x_raw = tensor.get_raw_data();

    let new_raw = x_raw.deref() + scalar;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}
