use crate::ops::public_ops::public_sub::{sub_tensor_scalar, sub_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::{Signed, Zero};
use std::fmt::Debug;
use std::ops::{Sub, SubAssign};

impl<'tl, TensorType, ScalarType> Sub<ScalarType> for &'tl Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<ScalarType, Output = TensorType> + Signed,
    ScalarType: SubAssign + ScalarOperand + DTypeMarker,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: ScalarType) -> Self::Output {
        return sub_tensor_scalar(self, rhs);
    }
}

impl<'tl_a, 'tl_b, TensorType> Sub<&'tl_b Tensor<TensorType>> for &'tl_a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<Output = TensorType> + Signed,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: &'tl_b Tensor<TensorType>) -> Self::Output {
        return sub_tensor_tensor(self, rhs);
    }
}

impl<'tl_a, TensorType> Sub<&'tl_a Tensor<TensorType>> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<Output = TensorType> + Signed,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: &'tl_a Tensor<TensorType>) -> Self::Output {
        return sub_tensor_tensor(&self, rhs);
    }
}
