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
