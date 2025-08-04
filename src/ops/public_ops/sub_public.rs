use ndarray::ScalarOperand;
use num_traits::Signed;
use std::fmt::Debug;
use std::ops::Sub;

use crate::ops::central::sub_impl::sub_impl;
use crate::ops::compute::sub_compute::{sub_compute_tensor_scalar, sub_compute_tensor_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn sub_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Sub<T> + Signed + 'static + Debug + Clone,
{
    let result_tensor = sub_compute_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from sub");
        sub_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor);
    }

    return result_tensor;
}

pub fn sub_tensor_scalar<T, S>(lhs_tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    T: DTComp + Sub<S, Output = T> + ScalarOperand + Signed + 'static + Debug + Clone,
    S: ScalarOperand,
{
    let result_tensor = sub_compute_tensor_scalar(lhs_tensor, scalar);

    if lhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from sub");
        sub_impl(Some(lhs_tensor), None, &result_tensor);
    }

    return result_tensor;
}
