use std::{iter::zip, ops::Add};

use crate::{
    Matrix,
    arith::{Scalar, TryAdd, TryMul},
    err::LinAlgError,
};

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

    use crate::arith::{try_add, try_mul};

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
}
