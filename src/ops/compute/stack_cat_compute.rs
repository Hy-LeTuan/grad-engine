use std::{fmt::Debug, ops::Deref};

use ndarray::{Axis, concatenate, stack};

use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};

pub fn stack_compute<T>(v: &[&Tensor<T>], dim: Axis) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
    let mut ref_array = vec![];
    let mut array_view_array = vec![];

    for tensor in v {
        let raw_array = tensor.get_raw_data();
        ref_array.push(raw_array);
    }

    for array_ref in &ref_array {
        array_view_array.push(array_ref.deref().view());
    }

    let stacked_res = stack(dim, &array_view_array);

    match stacked_res {
        Ok(stacked) => {
            let tensor = Tensor::from_raw_array(stacked, false);
            return tensor;
        }
        Err(e) => {
            panic!("Tensor Error: Cannot stack tensors. Error from {e}");
        }
    }
}

pub fn concatenate_compute<T>(v: &[&Tensor<T>], dim: Axis) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
    let mut ref_array = vec![];
    let mut array_view_array = vec![];

    for tensor in v {
        let raw_array = tensor.get_raw_data();
        ref_array.push(raw_array);
    }

    for array_ref in &ref_array {
        array_view_array.push(array_ref.deref().view());
    }

    let stacked_res = concatenate(dim, &array_view_array);

    match stacked_res {
        Ok(stacked) => {
            let tensor = Tensor::from_raw_array(stacked, false);
            return tensor;
        }
        Err(e) => {
            panic!("Tensor Error: Cannot concatenate tensors. Error from {e}");
        }
    }
}
