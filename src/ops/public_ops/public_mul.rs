use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Mul;

use crate::ops::central::mul_impl::mul_impl;
use crate::ops::compute::mul_compute::{compute_mul_tensor_scalar, compute_mul_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn mul_tensor_tensor<TensorType>(
    lhs_tensor: &Tensor<TensorType>,
    rhs_tensor: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    let result_tensor = compute_mul_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from mul");
    }

    mul_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor);

    return result_tensor;
}

pub fn mul_tensor_scalar<TensorType, ScalarType>(
    tensor: &Tensor<TensorType>,
    scalar: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Mul<ScalarType, Output = TensorType>
        + Mul<Output = TensorType>,
{
    let result_tensor = compute_mul_tensor_scalar(tensor, scalar);

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from mul");
    }

    mul_impl(Some(tensor), None, &result_tensor);

    return result_tensor;
}
