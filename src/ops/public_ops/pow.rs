use std::fmt::Debug;

use ndarray::ScalarOperand;
use num_traits::Float;

use crate::ops::public_ops::pow_public::pow_tensor;
use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};

impl<T> Tensor<T>
where
    T: DTComp + Debug + Float + 'static + ScalarOperand,
{
    pub fn pow(&self, base: T) -> Self {
        return pow_tensor(self, base);
    }
}
