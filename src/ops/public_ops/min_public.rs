use ndarray::Axis;
use num_traits::{Bounded, Float, Zero};
use std::fmt::Debug;
use std::rc::Rc;

use crate::ops::central::min_impl::min_impl;
use crate::ops::compute::max_min_compute::{argmin_compute_tensor, min_compute_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn min_tensor<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<T>
where
    T: Debug + DTComp + Clone + PartialOrd + Bounded + Zero + 'static + Float,
{
    if tensor.does_require_grad() {
        let (indices, new_tensor) = argmin_compute_tensor(tensor, dim, true);
        let new_tensor =
            new_tensor.expect("Internal error, no tensor found after calling min operation");
        new_tensor.requires_grad_intermediate("Intermediate tensor from min");

        min_impl(Some(tensor), &new_tensor, Rc::new(indices), dim);

        return new_tensor;
    } else {
        let new_tensor = min_compute_tensor(tensor, dim);
        return new_tensor;
    }
}

pub fn argmin_tensor<T>(tensor: &Tensor<T>, dim: Axis) -> Tensor<usize>
where
    T: Debug + DTComp + Clone + PartialOrd + Bounded + Zero + 'static + Float,
{
    let (indices, _) = argmin_compute_tensor(tensor, dim, false);
    return indices;
}
