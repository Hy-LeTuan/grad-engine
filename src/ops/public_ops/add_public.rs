use ndarray::ScalarOperand;
use std::fmt::Debug;
use std::ops::Add;

use crate::ops::central::add_impl::add_impl;
use crate::ops::compute::add_compute;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn add_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static,
{
    let result_tensor = add_compute::add_compute_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from add");
    }

    add_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor);

    return result_tensor;
}

pub fn add_tensor_scalar<T, S>(tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    T: DTComp + Clone + Add<S, Output = T> + Add<Output = T> + ScalarOperand + 'static + Debug,
    S: ScalarOperand,
{
    let result_tensor = add_compute::add_compute_tensor_scalar(tensor, scalar);

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from add");
    }

    add_impl(Some(tensor), None, &result_tensor);

    return result_tensor;
}
