use ndarray::LinalgScalar;
use std::fmt::Debug;
use std::ops::Add;

use crate::ops::public_ops::matmul_public::matmul_tensor_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn matmul_on_tensor<T>(lhs_tensor: Tensor<T>, rhs_tensor: Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static + LinalgScalar,
{
    return matmul(&lhs_tensor, &rhs_tensor);
}
pub fn matmul<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static + LinalgScalar,
{
    return matmul_tensor_tensor(lhs_tensor, rhs_tensor);
}
