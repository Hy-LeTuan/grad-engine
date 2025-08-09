use super::dtypes::{self, DTComp, DTypes};
use ndarray::{ArrayBase, Ix1, Ix2, IxDyn, OwnedRepr};

#[derive(Debug)]
pub struct Storage<T>
where
    T: DTComp,
{
    data: ArrayBase<OwnedRepr<T>, IxDyn>,
    nbytes: usize,
    dtype: dtypes::DTypes,
}

impl<T> Storage<T>
where
    T: DTComp,
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

impl<T> Storage<T>
where
    T: DTComp + Clone,
{
    pub fn get_data_as_ix2(&self) -> ArrayBase<OwnedRepr<T>, Ix2> {
        let fixed_shape = self
            .data
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("Error: Attempting to cast tensor as as 2D tensor failed");
        return fixed_shape;
    }

    pub fn get_data_as_ix1(&self) -> ArrayBase<OwnedRepr<T>, Ix1> {
        let fixed_shape = self
            .data
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("Error: Attempting to cast tensor as as 1D tensor failed");

        return fixed_shape;
    }
}
