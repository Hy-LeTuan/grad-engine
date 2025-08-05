use std::fmt::Debug;

use ndarray::ScalarOperand;
use num_traits::Float;

use crate::ops::public_ops::tanh_public::tanh_tensor;
use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};

impl<T> Tensor<T>
where
    T: DTComp + Debug + Float + 'static + ScalarOperand,
{
    pub fn tanh(&self) -> Self {
        return tanh_tensor(self);
    }
}
