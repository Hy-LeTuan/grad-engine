use std::fmt::Debug;
use std::ops::Add;

use crate::ops::central::broadcast_impl::broadcast_impl;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn broadcast_tensor<T>(tensor: &Tensor<T>, shape: Vec<usize>) -> Tensor<T>
where
    T: Debug + DTComp + Clone + Add<Output = T> + 'static,
{
    let raw_array = tensor.get_raw_data().to_owned();
    let new_array_option = raw_array.broadcast(shape);

    let result_tensor: Tensor<T> = match new_array_option {
        Some(new_array) => Tensor::from_raw_array(new_array.to_owned(), false),
        None => {
            panic!("Error: Cannot broadcast tensor to intended shape.");
        }
    };

    if tensor.does_require_grad() {
        result_tensor
            .requires_grad_intermediate("Intermediate tensor from broadcast with arbitrary base");

        broadcast_impl(Some(tensor), &result_tensor);
    }

    return result_tensor;
}
