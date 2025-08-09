pub fn get_shape_to_transpose_last_2_dim(original_shape: Vec<usize>) -> Vec<usize> {
    let intended_shape: Vec<usize> = (0..original_shape.len() - 2)
        .chain(vec![original_shape.len() - 1, original_shape.len() - 2])
        .collect();

    return intended_shape;
}

pub fn get_last_2_dim(shape: &[usize]) -> (usize, usize) {
    let last_2_dim = (shape[shape.len() - 2], shape[shape.len() - 1]);

    return last_2_dim;
}
