# Macro `tensor!`

## Source 

```rust
#[macro_export]
macro_rules! tensor {
    ( $([$([$([$([$([$($x:expr),* $(,)*]),+ $(,)*]),*]),*]),*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<Vec<Vec<_>>>>> = vec![$(vec![$(vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*])*])*])*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_6d(data, $requires_grad);

        tensor
    }};
    ( $([$([$([$([$($x:expr),* $(,)*]),+ $(,)*]),*]),*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<Vec<Vec<_>>>>> = vec![$(vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*])*])*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_5d(data, $requires_grad);

        tensor

    }};
    ( $([$([$([$($x:expr),* $(,)*]),+ $(,)*]),*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<Vec<_>>>> = vec![$(vec![$(vec![$(vec![$($x,)*],)*],)*])*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_4d(data, $requires_grad);

        tensor
    }};
    ( $([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data: Vec<Vec<Vec<_>>> = vec![$(vec![$(vec![$($x,)*],)*],)*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_3d(data, $requires_grad);

        tensor
    }};
    ( $([$($x:expr),*$(,)*]),+ $(,)*;requires_grad=$requires_grad:expr ) => {{
        let data : Vec<Vec<_>> = vec![$(vec![$($x,)*],)*];
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_2d(data, $requires_grad);

        tensor
    }};
    ( $($x:expr),*$(,)*;requires_grad=$requires_grad:expr ) => {{
        let tensor = $crate::tensor_core::tensor::Tensor::new_from_1d(vec![$($x,)*], $requires_grad);

        tensor
    }};
}
```

Create a `Tensor` with one, two, three, four, five, or six dimensions.

```rust
let t1 = tensor!(1, 2, 3, 4, 5; requires_grad=false);
let t2 = tensor!([1, 2, 3], [4, 5, 6]; requires_grad=false);
let t3 = tensor!([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]; requires_grad=true);
let t4 = tensor!([[[1, 2, 3, 4]]]; requires_grad=true);
let t5 = tensor!([[[[1, 2, 3, 4]]]]; requires_grad=true);
let t6 = tensor!([[[[[1, 2, 3, 4]]]]]; requires_grad=true);

assert_eq!(t1.get_shape().deref(), &[5]);
assert_eq!(t2.get_shape().deref(), &[2, 3]);
assert_eq!(t3.get_shape().deref(), &[2, 2, 3]);
assert_eq!(t4.get_shape().deref(), &[1, 1, 1, 4]);
assert_eq!(t5.get_shape().deref(), &[1, 1, 1, 1, 4]);
assert_eq!(t6.get_shape().deref(), &[1, 1, 1, 1, 1, 4]);
```

This macro utilizes the `vec!` macro and has the same ownership semantics; elements are moved into the resulting tensor. For any higher dimension array, use the `Tensor::new` syntax to create them.
