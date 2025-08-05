use num_traits::Signed;
use std::cell::{Ref, RefCell};
use std::fmt::Debug;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

pub fn neg_compute_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Signed,
{
    let raw_array = tensor.get_raw_data();
    let new_array = raw_array.map(|x| -x.clone());

    let tensor = Tensor::from_raw_array(new_array, false);

    return tensor;
}

pub fn neg_compute_tensorimpl<T>(tensorimpl: &RefCell<TensorImpl<T>>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Signed,
{
    let raw_array = Ref::map(tensorimpl.borrow(), |x| x.get_raw_data_());
    let new_array = raw_array.map(|x| -x.clone());

    let tensor = Tensor::from_raw_array(new_array, false);

    return tensor;
}
