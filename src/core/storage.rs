use crate::core::dtypes::DTypes;

use super::dtypes;
use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use num_traits::Zero;

pub struct Storage<F>
where
    F: Zero + Clone,
{
    data: ArrayBase<OwnedRepr<F>, IxDyn>,
    nbytes: usize,
    dtype: dtypes::DTypes,
}

impl<F> Storage<F>
where
    F: Zero + Clone,
{
    pub fn new(x: ArrayBase<OwnedRepr<F>, IxDyn>, nbytes: usize, dtype: dtypes::DTypes) -> Self {
        let storage = Storage {
            data: x,
            nbytes: nbytes,
            dtype: dtype,
        };

        return storage;
    }

    pub fn get_data(&self) -> &ArrayBase<OwnedRepr<F>, IxDyn> {
        return &(self.data);
    }

    pub fn get_nbytes(&self) -> usize {
        return self.nbytes;
    }

    pub fn get_dtype(self) -> DTypes {
        return self.dtype;
    }

    pub fn get_raw_shape(&self) -> Vec<usize> {
        return self.data.shape().to_vec();
    }
}
