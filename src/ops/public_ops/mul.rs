use crate::ops::public_ops::mul_public::{mul_tensor_scalar, mul_tensor_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use std::fmt::Debug;
use std::ops::{Add, Mul};

impl<'tl, T, S> Mul<S> for &'tl Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<Output = T> + Mul<S, Output = T> + Add<Output = T> + 'static,
    S: ScalarOperand,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: S) -> Self::Output {
        return mul_tensor_scalar(self, rhs);
    }
}

impl<'tl_in, 'tl_out, T> Mul<&'tl_out Tensor<T>> for &'tl_in Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<Output = T> + Add<Output = T> + 'static,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: &'tl_out Tensor<T>) -> Self::Output {
        return mul_tensor_tensor(self, rhs);
    }
}

impl<'tl, T> Mul<&'tl Tensor<T>> for Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<Output = T> + Add<Output = T> + 'static,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: &'tl Tensor<T>) -> Self::Output {
        return mul_tensor_tensor(&self, rhs);
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn mul_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);

        let _c = &a * &b;
    }

    #[test]
    fn mul_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = 3;

        let _c = &a * b;
    }
}
