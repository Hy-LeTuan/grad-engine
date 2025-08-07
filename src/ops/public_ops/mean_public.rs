use ndarray::{Axis, ScalarOperand};
use num_traits::NumCast;
use std::fmt::Debug;
use std::ops::{Add, Div};

use crate::ops::central::mean_impl::mean_impl;
use crate::ops::compute::sum_mean_compute::mean_compute_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn mean_tensor<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: Debug
        + DTComp
        + Clone
        + 'static
        + Add<Output = T>
        + NumCast
        + Div<Output = T>
        + ScalarOperand,
{
    let result_tensor = mean_compute_tensor(tensor, dim);

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from min");
        mean_impl(Some(tensor), &result_tensor, dim);
    }

    return result_tensor;
}
