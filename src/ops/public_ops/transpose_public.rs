use std::fmt::Debug;
use std::ops::Add;

use crate::ops::central::transpose_impl::transpose_impl;
use crate::ops::compute::shape_compute::compute_transpose;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn transpose_tensor<T>(tensor: &Tensor<T>, axes_option: Option<Vec<usize>>) -> Tensor<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    let result_tensor = compute_transpose(tensor, axes_option.clone());

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from min");
        transpose_impl(Some(tensor), &result_tensor, axes_option);
    }

    return result_tensor;
}
