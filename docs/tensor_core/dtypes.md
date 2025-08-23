# Dtypes

## Usage

The `Enum` defines all allowed types to be stored on the tensor. It also defines a way to convert from real Rust types to `Dtypes` through the `DTComp` trait.

## Definition

`DType` should be implemented at least for all types that `ndarray` supports. These should include:

```rust
#[derive(Debug, Copy, Clone)]
pub enum DTypes {
    Bool,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Isize,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Uint128,
    Usize,
}

```

# `DTComp` trait

This trait defines a method for raw Rust type to convert to `DTypes`. Only elements that implement this trait can be stored on a tensor.

## Definition

```rust
pub trait DTComp {
    fn dtype() -> DTypes;
}

impl DTComp for bool { }
impl DTComp for f32 { }
impl DTComp for f64 { }
impl DTComp for i8 { }
impl DTComp for i16 { }
impl DTComp for i32 { }
impl DTComp for i64 { }
impl DTComp for i128 { }
impl DTComp for isize { }
impl DTComp for u8 { }
impl DTComp for u16 { }
impl DTComp for u32 { }
impl DTComp for u64 { }
impl DTComp for u128 { }
impl DTComp for usize { }
```