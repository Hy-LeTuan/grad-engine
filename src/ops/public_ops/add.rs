use crate::ops::compute::add_compute::{compute_add_tensor_scalar, compute_add_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, AddAssign};

// ADD FOR TENSOR AND SCALAR

impl<TensorType, ScalarType> Add<ScalarType> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<ScalarType, Output = TensorType>,
    ScalarType: AddAssign + ScalarOperand + DTypeMarker,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: ScalarType) -> Tensor<TensorType> {
        return compute_add_tensor_scalar(&self, rhs);
    }
}

impl<TensorType> Add<Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_add_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Add<Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<f64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_add_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Add<Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_add_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Add<Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<i64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_add_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Add<Tensor<TensorType>> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Self) -> Self::Output {
        return compute_add_tensor_tensor(&self, &rhs);
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn add_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], false);

        let _c = a + b;
    }

    #[test]
    fn add_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = 3;

        let _c = a + b;
    }
}
