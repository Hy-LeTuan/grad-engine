use std::fmt::Debug;

use ndarray::ScalarOperand;
use num_traits::Float;

use crate::ops::public_ops::exp_public::exp_tensor;
use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};

impl<T> Tensor<T>
where
    T: DTComp + Debug + Float + 'static + ScalarOperand,
{
    pub fn exp(&self) -> Self {
        return exp_tensor(self, None);
    }

    pub fn exp2(&self) -> Self {
        return exp_tensor(self, Some(2));
    }
}
