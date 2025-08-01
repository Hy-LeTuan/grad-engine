use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Add;

use crate::ops::central::add_impl::add_impl;
use crate::ops::compute::add_compute;
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn add_tensor_tensor<TensorType>(
    lhs_tensor: &Tensor<TensorType>,
    rhs_tensor: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<TensorType>,
{
    let res = add_compute::compute_add_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        res.requires_grad();
    }

    add_impl(Some(lhs_tensor), Some(rhs_tensor), &res);

    return res;
}

pub fn add_tensor_scalar<TensorType, ScalarType>(
    tensor: &Tensor<TensorType>,
    scalar: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<ScalarType, Output = TensorType>,
{
    let res = add_compute::compute_add_tensor_scalar(tensor, scalar);

    if tensor.does_require_grad() {
        res.requires_grad()
    }

    add_impl(Some(tensor), None, &res);

    return res;
}
