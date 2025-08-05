use num_traits::Float;
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::Deref;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

pub fn tanh_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = tensor.get_raw_data();
    let raw_array = x_raw.mapv(|elem| elem.tanh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn tanh_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().mapv(|elem| elem.tanh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn sinh_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = tensor.get_raw_data();
    let raw_array = x_raw.mapv(|elem| elem.sinh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn sinh_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().mapv(|elem| elem.sinh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn cosh_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = tensor.get_raw_data();
    let raw_array = x_raw.mapv(|elem| elem.cosh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn cosh_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().mapv(|elem| elem.cosh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn sech_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = tensor.get_raw_data();
    let raw_array = x_raw.mapv(|elem| T::one() / elem.cosh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn sech_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().mapv(|elem| T::one() / elem.cosh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn csch_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = tensor.get_raw_data();
    let raw_array = x_raw.mapv(|elem| T::one() / elem.sinh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

pub fn csch_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float + 'static,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().mapv(|elem| T::one() / elem.sinh());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}
