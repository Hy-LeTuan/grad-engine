use ndarray::Axis;
use num_traits::{Bounded, Signed};
use std::fmt::Debug;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn min_compute<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Signed + Ord + Bounded,
{
    let raw_array = tensor.get_raw_data();
    let raw_array = raw_array.map_axis(dim, |view| {
        view.iter()
            .fold(T::max_value(), |acc, x| T::min(acc, x.clone()))
    });

    let tensor = Tensor::from_raw_array(raw_array, false);
    return tensor;
}

pub fn argmin_compute<T>(
    tensor: &Tensor<T>,
    dim: Axis,
    return_min: bool,
) -> (Tensor<usize>, Option<Tensor<T>>)
where
    T: DTComp + Clone + Debug + Signed + Ord + Bounded,
{
    let raw_array = tensor.get_raw_data();
    let mut output_shape = tensor.get_shape().to_vec();
    output_shape.remove(dim.index());

    let argmin_func = |(i, acc): (usize, T), (j, x): (usize, &T)| {
        if *x < acc {
            return (j, x.clone());
        } else {
            return (i, acc);
        }
    };

    let intermediate_pairs_representation = raw_array.map_axis(dim, |lane| {
        lane.iter()
            .enumerate()
            .fold((0, lane[0].clone()), argmin_func)
    });

    let argmin_index = intermediate_pairs_representation.map(|pair| pair.0.clone());
    let index_tensor = Tensor::from_raw_array(argmin_index, false);

    if return_min {
        let min_array = intermediate_pairs_representation.map(|pair| pair.1.clone());
        let min_tensor = Tensor::from_raw_array(min_array, false);

        return (index_tensor, Some(min_tensor));
    } else {
        return (index_tensor, None);
    }
}

pub fn max_compute<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Signed + Ord + Bounded,
{
    let raw_array = tensor.get_raw_data();
    raw_array.map_axis(Axis(0), |view| {
        view.iter()
            .fold(T::min_value(), |acc, x| T::max(acc, x.clone()))
    });

    todo!()
}

pub fn argmax_compute<T>(
    tensor: &Tensor<T>,
    dim: Axis,
    return_max: bool,
) -> (Tensor<usize>, Option<Tensor<T>>)
where
    T: DTComp + Clone + Debug + Signed + Ord + Bounded,
{
    let raw_array = tensor.get_raw_data();
    let mut output_shape = tensor.get_shape().to_vec();
    output_shape.remove(dim.index());

    let argmax_func = |(i, acc): (usize, T), (j, x): (usize, &T)| {
        if *x > acc {
            return (j, x.clone());
        } else {
            return (i, acc);
        }
    };

    let intermediate_pairs_representation = raw_array.map_axis(dim, |lane| {
        lane.iter()
            .enumerate()
            .fold((0, lane[0].clone()), argmax_func)
    });

    let argmax_index = intermediate_pairs_representation.map(|pair| pair.0.clone());
    let index_tensor = Tensor::from_raw_array(argmax_index, false);

    if return_max {
        let max_array = intermediate_pairs_representation.map(|pair| pair.1.clone());
        let max_tensor = Tensor::from_raw_array(max_array, false);

        return (index_tensor, Some(max_tensor));
    } else {
        return (index_tensor, None);
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn min_on_1d_array() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
        let _b = min_compute(&a, Axis(0));
    }

    #[test]
    fn argmin_on_2d_array() {
        let a = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true);
        let _b = argmin_compute(&a, Axis(0), false);
    }
}
