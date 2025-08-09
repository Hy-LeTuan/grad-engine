use ndarray::{ArrayD, Ix2, LinalgScalar, s};

use crate::{
    tensor_core::{dtypes::DTComp, tensor::Tensor},
    utils::shaping_utils::get_last_2_dim,
};
use std::{fmt::Debug, ops::Deref};

/// Matrix multiplication. If dimension is 2 or less, it acts as a normal dot product. For
/// multi-dimensional array, compute a batched matrix multiplication. **Caution**: For any matrix multiplication with differing batch dimension, you need to reshape / broadcast
/// them to match dimension first, since this function does not support implicit broadcasting  
pub fn matmul_compute_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + LinalgScalar,
{
    if lhs_tensor.get_shape().len() <= 2 && rhs_tensor.get_shape().len() <= 2 {
        return dot_compute_tensor_tensor(lhs_tensor, rhs_tensor);
    }

    let lhs_shape = lhs_tensor.get_shape();
    let rhs_shape = rhs_tensor.get_shape();

    let lhs_last_2_dim = get_last_2_dim(lhs_shape.deref());
    let rhs_last_2_dim = get_last_2_dim(rhs_shape.deref());

    if lhs_shape.len() != rhs_shape.len() {
        panic!(
            "Error: Batch matmul only supports tensors with the same number of dimensions. Got {:?} vs {:?}",
            lhs_shape, rhs_shape
        );
    }

    if lhs_last_2_dim.1 != rhs_last_2_dim.0 {
        panic!(
            "Error: Mismatching shape in dot product, lhs receive last 2 dim of shape: {:?} yet rhs receive last 2 dim of shape: {:?}",
            lhs_last_2_dim, rhs_last_2_dim
        );
    }

    let lhs_raw_array = lhs_tensor.get_raw_data();
    let rhs_raw_array = rhs_tensor.get_raw_data();

    let batch_dim = lhs_shape[..lhs_shape.len() - 2].to_vec();
    let flattened_batch_dim: usize = batch_dim.iter().product();

    let lhs_raw_array_batched = lhs_raw_array
        .broadcast(vec![
            flattened_batch_dim,
            lhs_last_2_dim.0,
            lhs_last_2_dim.1,
        ])
        .unwrap();

    let rhs_raw_array_batched = rhs_raw_array
        .broadcast(vec![
            flattened_batch_dim,
            rhs_last_2_dim.0,
            rhs_last_2_dim.1,
        ])
        .unwrap();

    // the final shape of the output array that needs to be reshaped to
    let mut final_output_shape = batch_dim.clone();
    final_output_shape.extend(vec![lhs_last_2_dim.0, rhs_last_2_dim.1]);

    let batched_output_shape = vec![flattened_batch_dim, lhs_last_2_dim.0, rhs_last_2_dim.1];
    let mut batched_output = ArrayD::<T>::zeros(batched_output_shape);

    // main computation loop
    for batch in 0..flattened_batch_dim {
        let lhs_slice = lhs_raw_array_batched
            .slice(s![batch, .., ..])
            .into_dimensionality::<Ix2>()
            .expect("Error: Internal error, cannot retrieve data from higher dimensional tensor");

        let rhs_slice = rhs_raw_array_batched
            .slice(s![batch, .., ..])
            .into_dimensionality::<Ix2>()
            .expect("Error: Internal error, cannot retrieve data from higher dimensional tensor");

        let z = lhs_slice.dot(&rhs_slice);
        batched_output.slice_mut(s![batch, .., ..]).assign(&z);
    }

    let final_output = batched_output
        .into_shape_with_order(final_output_shape)
        .expect("Error, cannot cast the final result of matrix multiplication into intended shape");

    let tensor = Tensor::from_raw_array(final_output, false);
    return tensor;
}

/// This is a subroutine of the matmul compute function, only used for tensors with dimensions of
/// at most 2
pub fn dot_compute_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + LinalgScalar,
{
    let lhs_rank = lhs_tensor.get_shape().len();
    let rhs_rank = rhs_tensor.get_shape().len();

    let result_raw_array = match (lhs_rank, rhs_rank) {
        (2, 2) => {
            let lhs = lhs_tensor.get_raw_data_as_ix2();
            let rhs = rhs_tensor.get_raw_data_as_ix2();
            lhs.dot(&rhs).into_dyn()
        }
        (2, 1) => {
            let lhs = lhs_tensor.get_raw_data_as_ix2();
            let rhs = rhs_tensor.get_raw_data_as_ix1();
            lhs.dot(&rhs).into_dyn()
        }
        (1, 2) => {
            let lhs = lhs_tensor.get_raw_data_as_ix1();
            let rhs = rhs_tensor.get_raw_data_as_ix2();
            lhs.dot(&rhs).into_dyn()
        }
        (1, 1) => {
            let lhs = lhs_tensor.get_raw_data_as_ix1();
            let rhs = rhs_tensor.get_raw_data_as_ix1();
            let scalar = lhs.dot(&rhs);
            ArrayD::from_shape_vec(vec![1], vec![scalar])
                .expect("Error: Failed to wrap scalar dot product in shape [1]")
        }
        _ => panic!("Error: dot_compute_tensor_tensor called with tensors of rank > 2"),
    };

    let tensor = Tensor::from_raw_array(result_raw_array, false);
    return tensor;
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    pub fn compute_dot_product_on_vector() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4], false).as_float_32();
        let b = Tensor::new(vec![1, 2, 3, 4], vec![4], false).as_float_32();

        let z = matmul_compute_tensor_tensor(&a, &b);

        assert_eq!(
            z.get_shape().len(),
            1usize,
            "Error: Output shape from tensor with 1 dimension has to result in a scalar"
        );

        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false).as_float_32();
        let b = Tensor::new(vec![1, 2, 3, 4], vec![1, 4], false).as_float_32();

        let _z = matmul_compute_tensor_tensor(&a, &b);
    }
}
