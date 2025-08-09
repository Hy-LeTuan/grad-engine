use std::fmt::Debug;
use std::ops::Add;

use crate::ops::central::reshape_impl::reshape_impl;
use crate::ops::compute::shape_compute::compute_reshape;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn reshape_tensor<T>(tensor: &Tensor<T>, axes_option: Vec<usize>) -> Tensor<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    let result_tensor = compute_reshape(tensor, axes_option.clone());

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from min");
        reshape_impl(Some(tensor), &result_tensor, tensor.get_shape().to_vec());
    }

    return result_tensor;
}
