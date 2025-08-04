use num_traits::Float;
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::Deref;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

pub fn log_compute_tensor<T>(lhs_scalar: &Tensor<T>, base: T) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = lhs_scalar.get_raw_data();

    let raw_array = x_raw.deref().log(base);

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn ln_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = tensor.get_raw_data();

    let raw_array = x_raw.deref().ln();

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn log_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>, base: T) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().log(base);

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn ln_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().ln();

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}
