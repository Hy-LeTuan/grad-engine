Packages needed:
1. [[Autograd Meta|AutogradMeta]]
2. [[Storage]]

```rust
struct Tensor {
	storage: Storage;
	sizes: usize;
	strides: usize;
	numel: i64;
	autograd_meta: Option<AutogradMeta>;
	version: u64;
}
```
