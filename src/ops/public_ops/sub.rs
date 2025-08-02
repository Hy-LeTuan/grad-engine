use crate::ops::public_ops::public_sub::{sub_tensor_scalar, sub_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Sub, SubAssign};

impl<'tl, TensorType, ScalarType> Sub<ScalarType> for &'tl Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<ScalarType, Output = TensorType>,
    ScalarType: SubAssign + ScalarOperand + DTypeMarker,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: ScalarType) -> Self::Output {
        return sub_tensor_scalar(self, rhs);
    }
}

impl<'tl_a, 'tl_b, TensorType> Sub<&'tl_b Tensor<TensorType>> for &'tl_a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: &'tl_b Tensor<TensorType>) -> Self::Output {
        return sub_tensor_tensor(self, rhs);
    }
}

impl<'tl_a, TensorType> Sub<&'tl_a Tensor<TensorType>> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: &'tl_a Tensor<TensorType>) -> Self::Output {
        return sub_tensor_tensor(&self, rhs);
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn subtract_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], false);
        let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], false);

        let _d = &a - &b - &c;
    }

    #[test]
    fn subtract_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false).as_float_32();
        let b = 4.0;
        let _c = &a - b;
    }
}
