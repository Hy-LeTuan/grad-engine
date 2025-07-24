use super::dtypes;
use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};
use num_traits::Zero;

pub struct Storage<F>
where
    F: Zero + Clone,
{
    data: ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>>,
    nbytes: usize,
    dtype: dtypes::DTypes,
}

impl<F> Storage<F>
where
    F: Zero + Clone,
{
    pub fn new(
        x: ArrayBase<OwnedRepr<F>, Dim<IxDynImpl>>,
        nbytes: usize,
        dtype: dtypes::DTypes,
    ) -> Self {
        let storage = Storage {
            data: x,
            nbytes: nbytes,
            dtype: dtype,
        };

        return storage;
    }
}
