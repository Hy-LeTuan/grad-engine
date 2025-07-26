use super::dtypes::{self, DTypes};
use ndarray::{ArrayBase, Dimension, Ix2, OwnedRepr};
use num_traits::Zero;

#[derive(Debug)]
pub struct Storage<T, D>
where
    T: Zero + Clone,
    D: Dimension,
{
    data: ArrayBase<OwnedRepr<T>, D>,
    nbytes: usize,
    dtype: dtypes::DTypes,
}

impl<T, D> Storage<T, D>
where
    T: Zero + Clone,
    D: Dimension,
{
    pub fn new(x: ArrayBase<OwnedRepr<T>, D>, nbytes: usize, dtype: dtypes::DTypes) -> Self {
        let storage = Storage {
            data: x,
            nbytes: nbytes,
            dtype: dtype,
        };

        return storage;
    }

    pub fn get_data(&self) -> &ArrayBase<OwnedRepr<T>, D> {
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
