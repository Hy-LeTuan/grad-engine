use ndarray::LinalgScalar;
use std::fmt::Debug;
use std::ops::Add;

use crate::ops::central::matmul_impl::matmul_impl;
use crate::ops::compute::matmul_compute::matmul_compute_tensor_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn matmul_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static + LinalgScalar,
{
    let result_tensor = matmul_compute_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from matmul operation");
    }

    matmul_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor);

    return result_tensor;
}
