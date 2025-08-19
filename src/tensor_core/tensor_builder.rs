use crate::tensor_core::{
    dtypes::DTComp, tensor::Tensor, tensor_impl::TensorImpl, tensor_utils::handle_requires_grad,
};
use ndarray::{ArrayD, IxDyn};
use num_traits::{AsPrimitive, One, Zero};

use std::{fmt::Debug, ops::Deref};

impl<T> Tensor<T>
where
    T: DTComp + Debug + AsPrimitive<f32>,
{
    pub fn as_float_32(&self) -> Tensor<f32> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        if self.does_require_grad() {
            tensor = Tensor::from_raw_array(new_raw_array, true);
        } else {
            tensor = Tensor::from_raw_array(new_raw_array, false);
        }

        return tensor;
    }

    pub fn ones_as_f32(shape: Vec<usize>) -> Tensor<f32> {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<f32>::ones(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor {
            tensor_impl: tensor_impl,
        };

        return tensor;
    }
}

impl<T> Tensor<T>
where
    T: DTComp + Debug + AsPrimitive<f64>,
{
    pub fn as_float_64(&self) -> Tensor<f64> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        if self.does_require_grad() {
            tensor = Tensor::from_raw_array(new_raw_array, true);
        } else {
            tensor = Tensor::from_raw_array(new_raw_array, false);
        }

        return tensor;
    }

    pub fn ones_as_f64(shape: Vec<usize>) -> Tensor<f64> {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<f64>::ones(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor { tensor_impl };

        return tensor;
    }
}

impl<T> Tensor<T>
where
    T: DTComp + Debug + Zero + Clone,
{
    pub fn zeros(shape: &Vec<usize>, requires_grad: Option<bool>) -> Self {
        let dyn_shape = IxDyn(shape);
        let data = ArrayD::<T>::zeros(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor { tensor_impl };

        handle_requires_grad(&tensor, requires_grad);

        return tensor;
    }

    pub fn zeros_like(tensor: &Tensor<T>, requires_grad: Option<bool>) -> Self {
        let shape = tensor.get_shape();
        return Tensor::zeros(shape.deref(), requires_grad);
    }
}

impl<T> Tensor<T>
where
    T: DTComp + Debug + Clone + One,
{
    pub fn ones_like(tensor: &Tensor<T>, requires_grad: Option<bool>) -> Self {
        let shape = tensor.get_shape();
        return Tensor::ones(shape.deref(), requires_grad);
    }

    pub fn ones(shape: &Vec<usize>, requires_grad: Option<bool>) -> Self {
        let dyn_shape = IxDyn(shape);
        let data = ArrayD::<T>::ones(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor { tensor_impl };

        handle_requires_grad(&tensor, requires_grad);

        return tensor;
    }
}
