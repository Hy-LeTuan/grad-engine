use ndarray::{Axis, ShapeArg};

use crate::tensor_core::{dtypes::DTComp, tensor::Tensor, tensor_impl::TensorImpl};
use std::{cell::RefCell, fmt::Debug};

pub fn compute_broadcast<T>(tensor: &Tensor<T>, shape: Vec<usize>) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
    let raw_array = tensor.get_raw_data();
    let new_array_option = raw_array.broadcast(shape);

    match new_array_option {
        Some(new_array) => {
            let new_tensor = Tensor::from_raw_array(new_array.to_owned(), false);

            return new_tensor;
        }
        None => {
            panic!("Error: Cannot broadcast the tensor to the intended shape.");
        }
    }
}

pub fn compute_unsqueeze<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
    let raw_array = tensor.get_raw_data().to_owned();
    let res_array = raw_array.insert_axis(dim);

    let tensor = Tensor::from_raw_array(res_array, false);

    return tensor;
}

pub fn compute_squeeze<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
    let raw_array = tensor.get_raw_data().to_owned();
    let res_array = raw_array.remove_axis(dim);

    let tensor = Tensor::from_raw_array(res_array, false);

    return tensor;
}

pub fn compute_reshape<T, E>(tensor: &Tensor<T>, shape: E) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
    E: ShapeArg + Debug + Clone,
{
    let raw_array = tensor.get_raw_data().to_owned();
    let res_array_result = raw_array.into_shape_with_order(shape);

    match res_array_result {
        Ok(res_array) => {
            let tensor = Tensor::from_raw_array(res_array.into_dyn(), false);
            return tensor;
        }
        Err(e) => {
            panic!("{:?}", e);
        }
    }
}

pub fn compute_reshape_tensorimpl<T, E>(tensorimpl: &RefCell<TensorImpl<T>>, shape: E) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
    E: ShapeArg + Debug + Clone,
{
    let raw_array = tensorimpl.borrow().get_raw_data_().to_owned();
    let res_array_result = raw_array.into_shape_with_order(shape);

    match res_array_result {
        Ok(res_array) => {
            let tensor = Tensor::from_raw_array(res_array.into_dyn(), false);
            return tensor;
        }
        Err(e) => {
            panic!("{:?}", e);
        }
    }
}

pub fn compute_transpose<T>(tensor: &Tensor<T>, axes_option: Option<Vec<usize>>) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
    let shape = tensor.get_shape();

    if shape.len() == 1 {
        panic!("Error, cannot transpose a 1D tensor");
    } else if shape.len() == 2 {
        let raw_array = tensor.get_raw_data_as_ix2();
        let res_array_result = raw_array.permuted_axes((1, 0)).into_dyn();

        let tensor = Tensor::from_raw_array(res_array_result, false);
        return tensor;
    } else {
        match axes_option {
            Some(axes) => {
                let raw_array = tensor.get_raw_data().to_owned();
                let res_array_result = raw_array.permuted_axes(axes).into_dyn();

                let tensor = Tensor::from_raw_array(res_array_result, false);
                return tensor;
            }
            None => {
                panic!(
                    "Error: Trying to transpose a multi-dimensional tensor with no axes order provided. Try passing in the argument for axes_option."
                );
            }
        }
    }
}

pub fn compute_transpose_tensorimpl<T, E>(
    tensorimpl: &RefCell<TensorImpl<T>>,
    axes_option: Option<Vec<usize>>,
) -> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
    let shape = tensorimpl.borrow().get_raw_shape();

    if shape.len() == 1 {
        panic!("Error, cannot transpose a 1D tensor");
    } else if shape.len() == 2 {
        let raw_array = tensorimpl.borrow().get_raw_data_as_ix2_();
        let res_array_result = raw_array.permuted_axes((1, 0)).into_dyn();

        let tensor = Tensor::from_raw_array(res_array_result, false);
        return tensor;
    } else {
        match axes_option {
            Some(axes) => {
                let raw_array = tensorimpl.borrow().get_raw_data_().to_owned();
                let res_array_result = raw_array.permuted_axes(axes).into_dyn();

                let tensor = Tensor::from_raw_array(res_array_result, false);
                return tensor;
            }
            None => {
                panic!(
                    "Error: Trying to transpose a multi-dimensional tensor with no axes order provided. Try passing in the argument for axes_option."
                );
            }
        }
    }
}
