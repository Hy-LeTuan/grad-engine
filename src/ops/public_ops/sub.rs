use crate::ops::public_ops::sub_public::{sub_tensor_scalar, sub_tensor_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Signed;
use std::fmt::Debug;
use std::ops::Sub;

impl<'tl, T, S> Sub<S> for &'tl Tensor<T>
where
    T: DTComp + Sub<S, Output = T> + ScalarOperand + Signed + 'static + Debug + Clone,
    S: ScalarOperand,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: S) -> Self::Output {
        return sub_tensor_scalar(self, rhs);
    }
}

impl<T, S> Sub<S> for Tensor<T>
where
    T: DTComp + Sub<S, Output = T> + ScalarOperand + Signed + 'static + Debug + Clone,
    S: ScalarOperand,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: S) -> Self::Output {
        return &self - rhs;
    }
}

impl<'tl_a, 'tl_b, T> Sub<&'tl_b Tensor<T>> for &'tl_a Tensor<T>
where
    T: DTComp + Sub<T> + Signed + 'static + Debug + Clone,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: &'tl_b Tensor<T>) -> Self::Output {
        return sub_tensor_tensor(self, rhs);
    }
}

impl<'tl_a, T> Sub<&'tl_a Tensor<T>> for Tensor<T>
where
    T: DTComp + Sub<T> + Signed + 'static + Debug + Clone,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: &'tl_a Tensor<T>) -> Self::Output {
        return sub_tensor_tensor(&self, rhs);
    }
}

impl<T> Sub<Tensor<T>> for Tensor<T>
where
    T: DTComp + Sub<T> + Signed + 'static + Debug + Clone,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Tensor<T>) -> Self::Output {
        return &self - &rhs;
    }
}
