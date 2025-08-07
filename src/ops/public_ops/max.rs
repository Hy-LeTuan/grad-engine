use ndarray::Axis;
use num_traits::{Bounded, Float, Zero};
use std::fmt::Debug;

use crate::ops::public_ops::max_public::{argmax_tensor, max_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + PartialOrd + Bounded + Zero + 'static + Float,
{
    pub fn max(&self, dim: Axis) -> Self {
        return max_tensor(self, dim);
    }

    pub fn argmax(&self, dim: Axis) -> Tensor<usize> {
        return argmax_tensor(self, dim);
    }
}
