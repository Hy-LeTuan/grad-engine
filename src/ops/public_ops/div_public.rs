use ndarray::ScalarOperand;
use num_traits::Signed;
use std::fmt::Debug;
use std::ops::{Add, Div};

use crate::ops::central::div_impl::div_impl;
use crate::ops::compute::div_compute::{div_compute_tensor_scalar, div_compute_tensor_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn div_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp
        + Clone
        + Debug
        + Div<Output = T>
        + Add<Output = T>
        + 'static
        + ScalarOperand
        + Signed,
{
    let result_tensor = div_compute_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from div");
    }

    div_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor, None);

    return result_tensor;
}

pub fn div_tensor_scalar<T, S>(tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    T: DTComp
        + Clone
        + Debug
        + Div<Output = T>
        + Div<S, Output = T>
        + Add<Output = T>
        + 'static
        + Signed,
    S: ScalarOperand + Debug + Clone,
{
    let result_tensor = div_compute_tensor_scalar(tensor, scalar.clone());

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from div");
    }

    div_impl(Some(tensor), None, &result_tensor, Some(scalar.clone()));

    return result_tensor;
}
