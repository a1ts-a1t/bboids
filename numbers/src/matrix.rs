use std::iter;

use num::Zero;

use crate::err::ValueError;

pub struct Matrix<T> {
    vec: Vec<T>,
    stride: usize,
}

impl<T: Zero> Matrix<T> {
    fn zeros<S: Into<(usize, usize)>>(shape: S) -> Self {
        let (rows, cols) = shape.into();
        let len = rows * cols;
        let vec = iter::repeat_with(|| T::zero()).take(len).collect();
        Matrix { vec, stride: cols }
    }
}

impl<T> Matrix<T> {
    pub fn try_from_iter<Inner: Iterator<Item = T>, Outer: Iterator<Item = Inner>>(
        mut value: Outer,
    ) -> Result<Self, ValueError> {
        let mut first = match value.next() {
            Some(v) => v,
            None => {
                return Ok(Matrix {
                    vec: Vec::new(),
                    stride: 0,
                });
            }
        };

        // find the stride while populating the vec
        let mut vec = Vec::new();
        let mut stride = 0usize;
        while let Some(x) = first.next() {
            vec.push(x);
            stride += 1;
        }

        // for the remaining inner iters, check their lengths
        // while populating the vec
        while let Some(mut v) = value.next() {
            let mut v_stride = 0usize;
            while let Some(x) = v.next() {
                vec.push(x);
                v_stride += 1;
            }

            if v_stride != stride {
                return Err(ValueError);
            }
        }

        Ok(Matrix { vec, stride })
    }
}
