`Storage` struct is the struct containing the raw data for every tensor. Even though operations are defined on the tensors themselves, the data will be queried from `Storage` first before any operations are completed.

Requirements:
1. [[Dtypes]]

```rust
pub struct Storage {
	data:;
	nbytes: usize;
	dtype: Dtype;
}
```