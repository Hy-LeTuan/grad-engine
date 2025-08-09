use itertools::Itertools;
use ndarray::{ArrayD, Ix2, IxDyn, LinalgScalar, SliceInfo, SliceInfoElem};

use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};
use std::fmt::Debug;

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

    let lhs_last_2_dim = (
        lhs_shape[lhs_shape.len() - 2],
        lhs_shape[lhs_shape.len() - 1],
    );

    let rhs_last_2_dim = (
        rhs_shape[rhs_shape.len() - 2],
        rhs_shape[rhs_shape.len() - 1],
    );

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

    let mut output_shape = batch_dim.clone();
    output_shape.extend(vec![lhs_last_2_dim.0, rhs_last_2_dim.1]);

    println!("output shape: {:?}", output_shape);

    let mut output = ArrayD::<T>::zeros(output_shape.clone());

    // main computation loop
    let batches = batch_dim
        .iter()
        .map(|dim| 0..*dim)
        .multi_cartesian_product();

    for batch in batches {
        let elems: Vec<SliceInfoElem> = batch
            .clone()
            .iter()
            .map(|&i| SliceInfoElem::Index(i as isize))
            .collect();

        let mut lhs_elems = elems.clone();

        lhs_elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(lhs_last_2_dim.0 as isize),
            step: 1,
        });

        lhs_elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(lhs_last_2_dim.1 as isize),
            step: 1,
        });

        let mut rhs_elems = elems.clone();

        rhs_elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(rhs_last_2_dim.0 as isize),
            step: 1,
        });

        rhs_elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(rhs_last_2_dim.1 as isize),
            step: 1,
        });

        let mut output_elems = elems;
        output_elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(output_shape[output_shape.len() - 2] as isize),
            step: 1,
        });

        output_elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(output_shape[output_shape.len() - 1] as isize),
            step: 1,
        });

        unsafe {
            let lhs_info = SliceInfo::<_, IxDyn, IxDyn>::new(lhs_elems).unwrap();
            let rhs_info = SliceInfo::<_, IxDyn, IxDyn>::new(rhs_elems).unwrap();
            let output_info = SliceInfo::<_, IxDyn, IxDyn>::new(output_elems).unwrap();

            let lhs_slice = lhs_raw_array
                .slice(lhs_info.as_ref())
                .into_dimensionality::<Ix2>()
                .expect(
                    "Error: Internal error, cannot retrieve data from higher dimensional tensor",
                );

            let rhs_slice = rhs_raw_array
                .slice(rhs_info.as_ref())
                .into_dimensionality::<Ix2>()
                .expect(
                    "Error: Internal error, cannot retrieve data from higher dimensional tensor",
                );

            let z = lhs_slice.dot(&rhs_slice);
            output.slice_mut(output_info).assign(&z);
        }
    }

    let tensor = Tensor::from_raw_array(output, false);
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
    use std::ops::Deref;

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

    #[test]
    pub fn compute_matmul_on_high_dimensional_matrix() {
        let a = Tensor::new(
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
            vec![3, 2, 3],
            false,
        )
        .as_float_32();

        let b = Tensor::new(
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            ],
            vec![3, 3, 4],
            false,
        )
        .as_float_32();

        let z = matmul_compute_tensor_tensor(&a, &b);

        assert_eq!(
            z.get_shape().deref(),
            &vec![3usize, 2usize, 4usize],
            "Error: Output shape from matrix multiplication does not match"
        );
    }
}
