use ndarray::ScalarOperand;
use std::fmt::Debug;
use std::ops::{Add, Mul};

use crate::ops::central::mul_impl::mul_impl;
use crate::ops::compute::mul_compute::{mul_compute_tensor_scalar, mul_compute_tensor_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn mul_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<Output = T> + Add<Output = T> + 'static + ScalarOperand,
{
    let result_tensor = mul_compute_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from mul");
    }

    mul_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor, None);

    return result_tensor;
}

pub fn mul_tensor_scalar<T, S>(tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<Output = T> + Mul<S, Output = T> + Add<Output = T> + 'static,
    S: ScalarOperand + Debug + Clone,
{
    let result_tensor = mul_compute_tensor_scalar(tensor, scalar.clone());

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from mul");
    }

    mul_impl(Some(tensor), None, &result_tensor, Some(scalar.clone()));

    return result_tensor;
}
