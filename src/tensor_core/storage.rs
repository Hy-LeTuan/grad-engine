use super::dtypes::{self, DTypes};
use ndarray::{ArrayBase, Ix2, IxDyn, OwnedRepr};
use num_traits::Zero;

#[derive(Debug)]
pub struct Storage<T>
where
    T: Zero + Clone,
{
    data: ArrayBase<OwnedRepr<T>, IxDyn>,
    nbytes: usize,
    dtype: dtypes::DTypes,
}

impl<T> Storage<T>
where
    T: Zero + Clone,
{
    pub fn new(x: ArrayBase<OwnedRepr<T>, IxDyn>, nbytes: usize, dtype: dtypes::DTypes) -> Self {
        let storage = Storage {
            data: x,
            nbytes: nbytes,
            dtype: dtype,
        };

        return storage;
    }

    pub fn get_data(&self) -> &ArrayBase<OwnedRepr<T>, IxDyn> {
        return &(self.data);
    }

    pub fn get_data_as_ix2(&self) -> ArrayBase<OwnedRepr<T>, Ix2> {
        let fixed_shape = self.data.clone().into_dimensionality::<Ix2>().unwrap();
        return fixed_shape;
    }

    pub fn get_nbytes(&self) -> usize {
        return self.nbytes;
    }

    pub fn get_dtype(&self) -> DTypes {
        return self.dtype;
    }

    pub fn get_raw_shape(&self) -> Vec<usize> {
        return self.data.shape().to_vec();
    }
}
