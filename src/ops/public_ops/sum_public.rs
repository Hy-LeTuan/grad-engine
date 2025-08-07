use ndarray::Axis;
use std::fmt::Debug;
use std::ops::Add;

use crate::ops::central::sum_impl::sum_impl;
use crate::ops::compute::sum_mean_compute::sum_compute_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn sum_tensor<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: Debug + DTComp + Clone + 'static + Add<Output = T>,
{
    let result_tensor = sum_compute_tensor(tensor, dim);

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from min");
        sum_impl(Some(tensor), &result_tensor, dim);
    }

    return result_tensor;
}
