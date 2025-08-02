use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Mul, Sub};

use crate::ops::central::sub_impl::sub_impl;
use crate::ops::compute::sub_compute::{compute_sub_tensor_scalar, compute_sub_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn sub_tensor_tensor<TensorType>(
    lhs_tensor: &Tensor<TensorType>,
    rhs_tensor: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Sub<Output = TensorType>
        + Mul<f32, Output = TensorType>,
{
    let result_tensor = compute_sub_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from sub");
        println!("Set grad for intermediate tensor");
    }

    sub_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor);

    return result_tensor;
}

pub fn sub_tensor_scalar<TensorType, ScalarType>(
    tensor: &Tensor<TensorType>,
    scalar: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Sub<ScalarType, Output = TensorType>
        + Mul<f32, Output = TensorType>,
{
    let result_tensor = compute_sub_tensor_scalar(tensor, scalar);

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from sub");
    }

    sub_impl(Some(tensor), None, &result_tensor);

    return result_tensor;
}
