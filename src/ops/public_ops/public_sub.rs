use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Sub;

use crate::ops::compute::sub_compute::{compute_sub_tensor_scalar, compute_sub_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn sub_tensor_tensor<TensorType>(
    tensor_lhs: &Tensor<TensorType>,
    tensor_rhs: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<Output = TensorType>,
{
    let res = compute_sub_tensor_tensor(tensor_lhs, tensor_rhs);

    return res;
}

pub fn sub_tensor_scalar<TensorType, ScalarType>(
    tensor: &Tensor<TensorType>,
    scalar: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<ScalarType, Output = TensorType>,
{
    let res = compute_sub_tensor_scalar(tensor, scalar);
    return res;
}
