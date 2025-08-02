pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;

use tensor_core::tensor::Tensor;

fn main() {
    let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
    let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true);
    let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true);

    let d = &a + &b;

    let e = &d + &c;

    e.display_autograd_meta();

    println!("------------");

    let f = &a + &b + &c;
    f.display_autograd_meta();

    let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
    let x2 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
    let x3 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);

    let z = &x1 + &x2 + &x3;

    z.display_autograd_meta();
}
