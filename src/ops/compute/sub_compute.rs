use ndarray::ScalarOperand;
use std::ops::{Deref, Sub};

use std::fmt::Debug;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn sub_compute_tensor_tensor<T>(tensor_lhs: &Tensor<T>, tensor_rhs: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Sub<Output = T> + Debug,
{
    let lhs_raw = tensor_lhs.get_raw_data();
    let rhs_raw = tensor_rhs.get_raw_data();

    let new_raw = lhs_raw.deref() - rhs_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn sub_compute_tensor_scalar<T, S>(tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    T: DTComp + Clone + Sub<S, Output = T> + ScalarOperand + Debug,
    S: ScalarOperand,
{
    let x_raw = tensor.get_raw_data();

    let new_raw = x_raw.deref() - scalar;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}
