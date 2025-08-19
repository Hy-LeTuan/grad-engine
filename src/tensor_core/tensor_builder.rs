use crate::tensor_core::{
    dtypes::DTComp, tensor::Tensor, tensor_impl::TensorImpl, tensor_utils::handle_requires_grad,
};
use ndarray::{ArrayD, IxDyn};
use num_traits::{AsPrimitive, One, Zero};

use std::{fmt::Debug, ops::Deref};

#[macro_export]
macro_rules! tensor {
    ( $([$([$([$([$([$($x:expr),* $(,)*]),+ $(,)*]),*]),*]),*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<Vec<Vec<_>>>>> = vec![$(vec![$(vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*])*])*])*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_6d(data, $requires_grad);

        tensor
    }};
    ( $([$([$([$([$($x:expr),* $(,)*]),+ $(,)*]),*]),*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<Vec<Vec<_>>>>> = vec![$(vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*])*])*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_5d(data, $requires_grad);

        tensor

    }};
    ( $([$([$([$($x:expr),* $(,)*]),+ $(,)*]),*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<Vec<_>>>> = vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*])*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_4d(data, $requires_grad);

        tensor
    }};
    ( $([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<_>>> = vec![$(vec![$(vec![$($x,)*],)*],)*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_3d(data, $requires_grad);

        tensor
    }};
    ( $([$($x:expr),*$(,)*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data : Vec<Vec<_>> = vec![$(vec![$($x,)*],)*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_2d(data, $requires_grad);

        tensor
    }};
    ( $($x:expr),*$(,)*;requires_grad=$requires_grad:expr ) => {{
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_1d(vec![$($x,)*], $requires_grad);

        tensor
    }};
}

impl<T> Tensor<T>
where
    T: DTComp + Debug,
{
    pub fn new_from_1d(x: Vec<T>, requires_grad: bool) -> Tensor<T> {
        let data_size: usize = x.len();

        let tensor = Tensor::new(x, vec![data_size], requires_grad);

        return tensor;
    }

    pub fn new_from_2d(x: Vec<Vec<T>>, requires_grad: bool) -> Tensor<T> {
        let batch_dim: usize = x.len();
        let repeat_dim = x[0].len();

        let flattend_x: Vec<T> = x.into_iter().flatten().collect();

        let tensor = Tensor::new(flattend_x, vec![batch_dim, repeat_dim], requires_grad);

        return tensor;
    }

    pub fn new_from_3d(x: Vec<Vec<Vec<T>>>, requires_grad: bool) -> Tensor<T> {
        let first_dim = x.len();
        let second_dim = x[0].len();
        let third_dim = x[0][0].len();

        let flattened_x: Vec<T> = x.into_iter().flatten().flatten().collect();

        let tensor = Tensor::new(
            flattened_x,
            vec![first_dim, second_dim, third_dim],
            requires_grad,
        );

        return tensor;
    }

    pub fn new_from_4d(x: Vec<Vec<Vec<Vec<T>>>>, requires_grad: bool) -> Tensor<T> {
        let first_dim = x.len();
        let second_dim = x[0].len();
        let third_dim = x[0][0].len();
        let fourth_dim = x[0][0][0].len();

        let flattened_x: Vec<T> = x.into_iter().flatten().flatten().flatten().collect();

        let tensor = Tensor::new(
            flattened_x,
            vec![first_dim, second_dim, third_dim, fourth_dim],
            requires_grad,
        );

        return tensor;
    }

    pub fn new_from_5d(x: Vec<Vec<Vec<Vec<Vec<T>>>>>, requires_grad: bool) -> Tensor<T> {
        let first_dim = x.len();
        let second_dim = x[0].len();
        let third_dim = x[0][0].len();
        let fourth_dim = x[0][0][0].len();
        let fifth_dim = x[0][0][0][0].len();

        let flattened_x: Vec<T> = x
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .collect();

        let tensor = Tensor::new(
            flattened_x,
            vec![first_dim, second_dim, third_dim, fourth_dim, fifth_dim],
            requires_grad,
        );

        return tensor;
    }

    pub fn new_from_6d(x: Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>, requires_grad: bool) -> Tensor<T> {
        let first_dim = x.len();
        let second_dim = x[0].len();
        let third_dim = x[0][0].len();
        let fourth_dim = x[0][0][0].len();
        let fifth_dim = x[0][0][0][0].len();
        let sixth_dim = x[0][0][0][0][0].len();

        let flattened_x: Vec<T> = x
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .collect();

        let tensor = Tensor::new(
            flattened_x,
            vec![
                first_dim, second_dim, third_dim, fourth_dim, fifth_dim, sixth_dim,
            ],
            requires_grad,
        );

        return tensor;
    }
}

impl<T> Tensor<T>
where
    T: DTComp + Debug + AsPrimitive<f32>,
{
    pub fn as_float_32(&self) -> Tensor<f32> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        if self.does_require_grad() {
            tensor = Tensor::from_raw_array(new_raw_array, true);
        } else {
            tensor = Tensor::from_raw_array(new_raw_array, false);
        }

        return tensor;
    }

    pub fn ones_as_f32(shape: Vec<usize>) -> Tensor<f32> {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<f32>::ones(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor {
            tensor_impl: tensor_impl,
        };

        return tensor;
    }
}

impl<T> Tensor<T>
where
    T: DTComp + Debug + AsPrimitive<f64>,
{
    pub fn as_float_64(&self) -> Tensor<f64> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        if self.does_require_grad() {
            tensor = Tensor::from_raw_array(new_raw_array, true);
        } else {
            tensor = Tensor::from_raw_array(new_raw_array, false);
        }

        return tensor;
    }

    pub fn ones_as_f64(shape: Vec<usize>) -> Tensor<f64> {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<f64>::ones(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor { tensor_impl };

        return tensor;
    }
}

impl<T> Tensor<T>
where
    T: DTComp + Debug + Zero + Clone,
{
    pub fn zeros(shape: &Vec<usize>, requires_grad: Option<bool>) -> Self {
        let dyn_shape = IxDyn(shape);
        let data = ArrayD::<T>::zeros(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor { tensor_impl };

        handle_requires_grad(&tensor, requires_grad);

        return tensor;
    }

    pub fn zeros_like(tensor: &Tensor<T>, requires_grad: Option<bool>) -> Self {
        let shape = tensor.get_shape();
        return Tensor::zeros(shape.deref(), requires_grad);
    }
}

impl<T> Tensor<T>
where
    T: DTComp + Debug + Clone + One,
{
    pub fn ones_like(tensor: &Tensor<T>, requires_grad: Option<bool>) -> Self {
        let shape = tensor.get_shape();
        return Tensor::ones(shape.deref(), requires_grad);
    }

    pub fn ones(shape: &Vec<usize>, requires_grad: Option<bool>) -> Self {
        let dyn_shape = IxDyn(shape);
        let data = ArrayD::<T>::ones(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor { tensor_impl };

        handle_requires_grad(&tensor, requires_grad);

        return tensor;
    }
}

#[cfg(test)]
pub mod test {
    use crate::utils::testing_utils;

    #[allow(unused)]
    use super::*;

    #[test]
    fn test_macro_on_1d() {
        let tensor = tensor!(1, 2, 3, 4, 5; requires_grad=false);
        let target = Tensor::new(vec![1, 2, 3, 4, 5], vec![5], false);

        testing_utils::epsilon_test_for_tensor_similarity(
            tensor.get_raw_data(),
            target.get_raw_data(),
            1e-4,
        );
    }

    #[test]
    fn test_macro_on_2d() {
        let tensor = tensor!([1, 2, 3], [4, 5, 6]; requires_grad=false);
        let target = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], false);

        testing_utils::epsilon_test_for_tensor_similarity(
            tensor.get_raw_data(),
            target.get_raw_data(),
            1e-4,
        );
    }

    #[test]
    fn test_macro_on_3d() {
        let tensor = tensor!([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]; requires_grad=true);
        let target = Tensor::new(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            vec![2, 2, 3],
            false,
        );

        testing_utils::epsilon_test_for_tensor_similarity(
            tensor.get_raw_data(),
            target.get_raw_data(),
            1e-4,
        );
    }

    #[test]
    fn test_macro_on_4d() {
        let tensor = tensor!([[[1, 2, 3, 4]]]; requires_grad=true);
        let target = Tensor::new(vec![1, 2, 3, 4], vec![1, 1, 1, 4], false);

        testing_utils::epsilon_test_for_tensor_similarity(
            tensor.get_raw_data(),
            target.get_raw_data(),
            1e-4,
        );
    }

    #[test]
    fn test_macro_on_5d() {
        let tensor = tensor!([[[[1, 2, 3, 4]]]]; requires_grad=true);
        let target = Tensor::new(vec![1, 2, 3, 4], vec![1, 1, 1, 1, 4], false);

        testing_utils::epsilon_test_for_tensor_similarity(
            tensor.get_raw_data(),
            target.get_raw_data(),
            1e-4,
        );
    }

    #[test]
    fn test_macro_on_6d() {
        let tensor = tensor!([[[[[1, 2, 3, 4]]]]]; requires_grad=true);
        let target = Tensor::new(vec![1, 2, 3, 4], vec![1, 1, 1, 1, 1, 4], false);

        testing_utils::epsilon_test_for_tensor_similarity(
            tensor.get_raw_data(),
            target.get_raw_data(),
            1e-4,
        );
    }
}
