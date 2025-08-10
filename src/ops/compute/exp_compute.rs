use num_traits::Float;
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::Deref;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

/// Compute the exponential of the tensor with base e
pub fn exp_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float,
{
    let x_raw = tensor.get_raw_data();
    let raw_array = x_raw.mapv(|elem| elem.exp());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

/// Compute the exponential of the tensorimpl with base e
pub fn exp_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().mapv(|elem| elem.exp());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

/// Compute the exponential of the tensor with base 2
pub fn exp2_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float,
{
    let x_raw = tensor.get_raw_data();
    let raw_array = x_raw.mapv(|elem| elem.exp2());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

/// Compute the exponential of the tensorimpl with base e
pub fn exp2_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Float,
{
    let x_raw = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let raw_array = x_raw.deref().mapv(|elem| elem.exp2());

    let tensor = Tensor::from_raw_array(raw_array, false);

    return tensor;
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn compute_exp() {
        let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let z = exp_compute_tensor(&x1);

        println!("z is: {:?}", z);
    }
}
