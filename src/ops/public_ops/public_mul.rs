use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Mul;

use crate::ops::compute::mul_compute::{compute_mul_tensor_scalar, compute_mul_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn mul_tensor_tensor<TensorType>(
    lhs_scalar: &Tensor<TensorType>,
    rhs_tensor: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    let res = compute_mul_tensor_tensor(lhs_scalar, rhs_tensor);
    return res;
}

pub fn mul_tensor_scalar<TensorType, ScalarType>(
    tensor: &Tensor<TensorType>,
    scalar: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<ScalarType, Output = TensorType>,
{
    let res = compute_mul_tensor_scalar(tensor, scalar);
    return res;
}
