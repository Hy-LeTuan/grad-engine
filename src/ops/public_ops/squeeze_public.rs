use std::fmt::Debug;
use std::ops::Add;

use ndarray::Axis;

use crate::ops::central::squeeze_impl::squeeze_impl;
use crate::ops::compute::shape_compute::compute_squeeze;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn squeeze_tensor<T>(tensor: &Tensor<T>, reduced_dim: Axis) -> Tensor<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    let result_tensor = compute_squeeze(tensor, Axis(reduced_dim.index()));

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from min");
        squeeze_impl(Some(tensor), &result_tensor, reduced_dim);
    }

    return result_tensor;
}
