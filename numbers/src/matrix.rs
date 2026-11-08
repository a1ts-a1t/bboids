use std::{
    iter::{self, zip},
    ops::{Add, Index},
};

use num::{One, Zero};

use crate::{
    arith::{Scalar, TryAdd, TryMul},
    err::{LinAlgError, ValueError},
};

#[derive(PartialEq, Debug)]
pub struct Matrix<T> {
    vec: Vec<T>,
    stride: usize,
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.vec[row * self.stride + col]
    }
}

impl<T> Matrix<T> {
    pub fn shape(&self) -> (usize, usize) {
        if self.stride == 0 {
            return (0, 0);
        }

        let len = self.vec.len();
        (len / self.stride, self.stride)
    }

    pub fn is_square(&self) -> bool {
        let shape = self.shape();
        shape.0 == shape.1
    }

    pub fn rows(&self) -> Vec<&[T]> {
        self.vec.chunks(self.stride).collect()
    }
}

impl<T: Clone> Matrix<T> {
    pub fn transposed(&self) -> Self {
        let (m, n) = self.shape();

        let mut vec = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                vec.push(self[(i, j)].clone());
            }
        }

        Matrix { vec, stride: m }
    }

    pub fn transpose(&mut self) {
        if self.is_square() {
            let (n, _) = self.shape();

            for i in 0..n {
                for j in i..n {
                    self.vec.swap(i * n + j, j * n + i);
                }
            }

            return;
        }

        // i didn't realize the rabbithole this was........
        // i'm not gonna bother but further reading here
        // https://en.wikipedia.org/wiki/In-place_matrix_transposition#Algorithms
        *self = self.transposed();
    }
}

impl<T: One + Zero> Matrix<T> {
    pub fn identity(n: usize) -> Self {
        let mut vec = Vec::new();
        for i in 0..n * n {
            if i % (n + 1) == 0 {
                vec.push(T::one());
            } else {
                vec.push(T::zero());
            }
        }

        Matrix { vec, stride: n }
    }
}

impl<T: Zero> Matrix<T> {
    pub fn zeros<S: Into<(usize, usize)>>(shape: S) -> Self {
        let (rows, cols) = shape.into();
        let len = rows * cols;
        let vec = iter::repeat_with(|| T::zero()).take(len).collect();
        Matrix { vec, stride: cols }
    }
}

impl<T> Matrix<T> {
    pub fn try_from_iter<Inner: IntoIterator<Item = T>, Outer: IntoIterator<Item = Inner>>(
        value: Outer,
    ) -> Result<Self, ValueError> {
        let mut value_iter = value.into_iter();
        let first = match value_iter.next() {
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
        let first_iter = first.into_iter();
        for x in first_iter {
            vec.push(x);
            stride += 1;
        }

        // for the remaining inner iters, check their lengths
        // while populating the vec
        for v in value_iter {
            let mut v_stride = 0usize;
            let v_iter = v.into_iter();
            for x in v_iter {
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

impl<T: Add<Output = T>> TryAdd for Matrix<T> {
    type Error = LinAlgError;
    type Output = Self;

    fn try_add(self, other: Self) -> Result<Self::Output, Self::Error> {
        if self.shape() != other.shape() {
            return Err(LinAlgError::IncompatibleShape);
        }

        let sum_vec = zip(self.vec, other.vec).map(|(a, b)| a + b).collect();

        Ok(Matrix {
            vec: sum_vec,
            stride: self.stride,
        })
    }
}

impl<T> TryAdd for &Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Error = LinAlgError;
    type Output = Matrix<T>;

    fn try_add(self, other: Self) -> Result<Self::Output, Self::Error> {
        if self.shape() != other.shape() {
            return Err(LinAlgError::IncompatibleShape);
        }

        let sum_vec = zip(self.vec.iter(), other.vec.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Ok(Matrix {
            vec: sum_vec,
            stride: self.stride,
        })
    }
}

impl<T: Scalar + Clone> TryMul<Matrix<T>> for Matrix<T> {
    type Error = LinAlgError;
    type Output = Matrix<T>;

    fn try_mul(self, other: Matrix<T>) -> Result<Self::Output, Self::Error> {
        let (m, k) = self.shape();
        let (k_b, n) = other.shape();

        if k != k_b {
            return Err(LinAlgError::IncompatibleShape);
        }

        // bro, you are not getting a sub-cubic algorithm here
        // this is taken from lapack/blas's dgemm.f
        // https://netlib.org/lapack/lapack-3.1.1/html/dgemm.f.html
        let mut product = vec![T::zero(); m * n];
        for i in 0..m {
            for l in 0..k {
                let a_il = self[(i, l)].clone();
                for j in 0..n {
                    product[i * n + j] =
                        product[i * n + j].clone() + a_il.clone() * other[(l, j)].clone();
                }
            }
        }

        Ok(Matrix {
            vec: product,
            stride: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arith::{try_add, try_mul},
        err::ValueError,
    };

    #[test]
    fn test_zeros() {
        let expected: Matrix<f64> = Matrix {
            vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].to_vec(),
            stride: 3,
        };
        let actual: Matrix<f64> = Matrix::zeros((2, 3));
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_try_from_iter() {
        let expected: Matrix<f64> = Matrix {
            vec: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_vec(),
            stride: 3,
        };
        let actual: Matrix<f64> =
            Matrix::try_from_iter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_try_from_iter_fails() {
        let expected = ValueError;
        let actual = Matrix::try_from_iter([[1, 2, 3].to_vec(), [4, 5].to_vec()]).unwrap_err();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_try_add() {
        let expected: Matrix<f64> = Matrix {
            vec: [3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_vec(),
            stride: 3,
        };

        let a = Matrix::try_from_iter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        let b = Matrix::try_from_iter([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]).unwrap();
        let actual = a.try_add(b).unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_try_add_fails() {
        let expected = LinAlgError::IncompatibleShape;

        let a = Matrix::try_from_iter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        let b = Matrix::try_from_iter([[2.0, 2.0], [2.0, 2.0]]).unwrap();
        let actual = try_add(&a, &b).unwrap_err();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_shape() {
        let expected = (2, 3);

        let m = Matrix::try_from_iter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        let actual = m.shape();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_identity() {
        let expected = Matrix::try_from_iter([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).unwrap();
        let actual: Matrix<i32> = Matrix::identity(3);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_try_mul_matrix() {
        let expected = Matrix::try_from_iter([[58.0, 64.0], [139.0, 154.0]]).unwrap();
        let a = Matrix::try_from_iter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        let b = Matrix::try_from_iter([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]).unwrap();
        let actual = try_mul(a, b).unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_try_mul_matrix_fails() {
        let expected = LinAlgError::IncompatibleShape;
        let a = Matrix::try_from_iter([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        let b = Matrix::try_from_iter([[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let actual = try_mul(a, b).unwrap_err();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_index() {
        let expected = 3;
        let a = Matrix::try_from_iter([[1, 4], [2, 5], [3, 6]]).unwrap();
        let actual = a[(2, 0)];

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_transposed() {
        let expected = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6]]).unwrap();
        let a = Matrix::try_from_iter([[1, 4], [2, 5], [3, 6]]).unwrap();
        let actual = a.transposed();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_transpose_square() {
        let expected = Matrix::try_from_iter([[1, 2], [3, 4]]).unwrap();
        let mut actual = Matrix::try_from_iter([[1, 3], [2, 4]]).unwrap();
        actual.transpose();

        assert_eq!(expected, actual);
    }
}
