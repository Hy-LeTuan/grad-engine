# Ops

## Usage

The `{ops}.rs` files are the files that implement the overrides of pure Rust functions such as `ops::Add`, `ops::Sub`...

These endpoints then call functions defined in `{function}_public.rs` where the engine implements the core public interface for tensor-tensor and tensor-scalar computation. These endpoints also handle the computation graph creation and integral structure. The endpoints are explained in [[ops_public]].

## Sample

```rust
use crate::ops::public_ops::add_public::{add_tensor_scalar};

impl<'tl, T, S> Add<S> for &'tl Tensor<T>
where
    T: DTComp + Clone + Add<Output = T> + Add<S, Output = T> + ScalarOperand + 'static + Debug,
    S: ScalarOperand,
{
    type Output = Tensor<T>;

    fn add(self, rhs: S) -> Tensor<T> {
        return add_tensor_scalar(self, rhs);
    }
}
```