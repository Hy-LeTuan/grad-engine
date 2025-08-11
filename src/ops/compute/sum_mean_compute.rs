use ndarray::{Axis, ScalarOperand};
use num_traits::NumCast;
use std::{
    fmt::Debug,
    ops::{Add, Div},
};

use crate::{
    ops::compute::div_compute::div_compute_tensor_scalar,
    tensor_core::{dtypes::DTComp, tensor::Tensor},
};

pub fn sum_compute_tensor<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T>,
{
    let raw_array = tensor.get_raw_data();
    let raw_array = raw_array.map_axis(dim, |view| {
        view.iter()
            .cloned()
            .reduce(|a, b| a + b)
            .expect("Cannot compute sum over an empty axis")
    });

    let tensor = Tensor::from_raw_array(raw_array, false);
    return tensor;
}

pub fn sum_to_size_compute_tensor<T>(tensor: &Tensor<T>, intended_shape: &[usize]) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static,
{
    let tensor_shape = tensor.get_shape().to_vec();

    let mut result_tensor = Tensor::from_raw_array(tensor.get_raw_data().to_owned(), false);

    if tensor.get_shape().len() < intended_shape.len() {
        panic!(
            "Error: Cannot call sum to size on a size that is smaller than the tensor's current size"
        );
    }

    // remove leading dimension
    if tensor.get_shape().len() > intended_shape.len() {
        let dim_diff = tensor.get_shape().len() - intended_shape.len();
        for _ in 0..dim_diff {
            result_tensor = result_tensor.sum(Axis(0));
        }
    }

    // assuming they now have the same shape
    for ((index, curr_dim), intended_dim) in
        tensor_shape.iter().enumerate().zip(intended_shape.iter())
    {
        if curr_dim > intended_dim && *intended_dim == 1 {
            result_tensor = result_tensor.sum(Axis(index));
            result_tensor = result_tensor.unsqueeze(Axis(index));
        }
    }

    return result_tensor;
}

pub fn mean_compute_tensor<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + Div<Output = T> + ScalarOperand + NumCast,
{
    let num_elem = tensor.get_shape()[dim.index()];
    let num_elem = T::from(num_elem).expect("Failed to cast usize to target type");

    let raw_array = tensor.get_raw_data();
    let raw_array = raw_array.map_axis(dim, |view| {
        view.iter()
            .cloned()
            .reduce(|a, b| a + b)
            .expect("Cannot compute mean over an empty axis")
    });

    let tensor = Tensor::from_raw_array(raw_array, false);
    return div_compute_tensor_scalar(&tensor, num_elem);
}
