use crate::ops::public_ops::public_add::{add_tensor_scalar, add_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, AddAssign};

// ADD FOR TENSOR AND SCALAR

impl<'tl, TensorType, ScalarType> Add<ScalarType> for &'tl Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<ScalarType, Output = TensorType>,
    ScalarType: AddAssign + ScalarOperand + DTypeMarker,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: ScalarType) -> Tensor<TensorType> {
        return add_tensor_scalar(self, rhs);
    }
}

impl<'tl, TensorType> Add<&'tl Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return add_tensor_scalar(rhs, self);
    }
}

impl<'tl, TensorType> Add<&'tl Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<f64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return add_tensor_scalar(rhs, self);
    }
}

impl<'tl, TensorType> Add<&'tl Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return add_tensor_scalar(rhs, self);
    }
}

impl<'tl, TensorType> Add<&'tl Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<i64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return add_tensor_scalar(rhs, self);
    }
}

impl<'tl_a, 'tl_b, TensorType> Add<&'tl_b Tensor<TensorType>> for &'tl_a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'tl_b Tensor<TensorType>) -> Self::Output {
        return add_tensor_tensor(self, rhs);
    }
}

// allow the chaining of Add operation
impl<'tl, TensorType> Add<&'tl Tensor<TensorType>> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return add_tensor_tensor(&self, rhs);
    }
}
