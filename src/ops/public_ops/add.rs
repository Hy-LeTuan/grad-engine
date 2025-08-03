use crate::ops::public_ops::add_public::{add_tensor_scalar, add_tensor_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use std::fmt::Debug;
use std::ops::Add;

// ADD FOR TENSOR AND SCALAR

impl<'tl, T, S> Add<S> for &'tl Tensor<T>
where
    T: DTComp + Clone + Add<Output = T> + Add<S, Output = T> + ScalarOperand + 'static + Debug,
    S: ScalarOperand,
{
    type Output = Tensor<T>;

    fn add(self, rhs: S) -> Tensor<T> {
        return add_tensor_scalar(self, rhs);
    }
}

impl<'tl_in, 'tl_out, T> Add<&'tl_out Tensor<T>> for &'tl_in Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static,
{
    type Output = Tensor<T>;

    fn add(self, rhs: &'tl_out Tensor<T>) -> Self::Output {
        return add_tensor_tensor(self, rhs);
    }
}

// allow the chaining of Add operation
impl<'tl, T> Add<&'tl Tensor<T>> for Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static,
{
    type Output = Tensor<T>;

    fn add(self, rhs: &'tl Tensor<T>) -> Self::Output {
        return add_tensor_tensor(&self, rhs);
    }
}
