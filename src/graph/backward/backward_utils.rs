use crate::{
    ops::compute::sum_mean_compute::sum_to_size_compute_tensor,
    tensor_core::{dtypes::DTComp, tensor::Tensor},
};
use std::{fmt::Debug, ops::Add};

pub fn gradient_from_broadcast<T>(tensor: &Tensor<T>, intended_shape: &[usize]) -> Tensor<T>
where
    T: DTComp + Debug + Clone + Add<Output = T> + 'static,
{
    return sum_to_size_compute_tensor(tensor, intended_shape);
}

// #[cfg(test)}]
// pub mod test {
//     #[allow(unused)]
//     use super::*;
//
//     #[test]
//     fn test_gradient_from_broadcast() {
//         let grad = Tensor::new(
//             vec![
//                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                 1, 1, 1, 1, 1, 1, 1, 1,
//             ],
//             vec![3, 3, 4],
//             false,
//         );
//
//         let new_grad = gradient_from_broadcast(grad, &vec![4]);
//     }
// }
