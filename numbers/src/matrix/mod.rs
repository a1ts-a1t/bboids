mod arith;
mod index;
mod view;

pub use index::MatrixIndex;
pub use view::MatrixView;

use std::{
    iter,
    ops::{Index, IndexMut},
};

use num::{One, Zero};

use crate::{Vector, err::ValueError};

#[derive(PartialEq, Debug)]
pub struct Matrix<T> {
    pub(crate) vec: Vec<T>,
    pub(crate) stride: usize,
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.vec[row * self.stride + col]
    }
}

impl<T> From<Vector<T>> for Matrix<T> {
    fn from(value: Vector<T>) -> Self {
        let vec = value.into_iter().collect();
        Matrix { vec, stride: 1 }
    }
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

    pub fn view<'a, I: MatrixIndex>(&'a self, index: I) -> MatrixView<'a, T> {
        let shape = self.shape();
        MatrixView {
            buffer: &self.vec,
            stride: self.stride,
            offset: index.offset(shape),
            rows: index.rows(shape),
            columns: index.columns(shape),
        }
    }

    pub fn checked_view<'a, I: MatrixIndex>(&'a self, index: I) -> Option<MatrixView<'a, T>> {
        let shape = self.shape();
        let offset = index.offset(shape);
        let rows = index.rows(shape);
        let columns = index.columns(shape);

        let (m, n) = shape;
        if rows > m || columns > n || offset >= m * n {
            return None;
        }

        Some(MatrixView {
            buffer: &self.vec,
            stride: self.stride,
            offset,
            rows,
            columns,
        })
    }

    pub fn row<'a>(&'a self, i: usize) -> MatrixView<'a, T> {
        let (_, n) = self.shape();
        let offset = i * self.stride;

        MatrixView {
            buffer: &self.vec,
            stride: self.stride,
            offset,
            rows: 1,
            columns: n,
        }
    }

    pub fn column(&self, j: usize) -> impl Iterator<Item = &T> {
        self.vec.iter().skip(j).step_by(self.stride)
    }
}

impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Matrix {
            vec: self.vec.clone(),
            stride: self.stride,
        }
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

impl<T: Zero> Matrix<T> {
    pub fn zeros<S: Into<(usize, usize)>>(shape: S) -> Self {
        let (rows, cols) = shape.into();
        let len = rows * cols;
        let vec = iter::repeat_with(|| T::zero()).take(len).collect();
        Matrix { vec, stride: cols }
    }
}

impl<T: Zero + Clone> Matrix<T> {
    pub fn triu(&self, offset: isize) -> Self {
        let mut vec = [].to_vec();
        for (idx, val) in self.vec.iter().enumerate() {
            let i = idx / self.stride;
            let j = idx % self.stride;

            if i.saturating_add_signed(offset) <= j {
                vec.push(val.clone());
            } else {
                vec.push(T::zero());
            }
        }

        Matrix {
            vec,
            stride: self.stride,
        }
    }

    pub fn tril(&self, offset: isize) -> Self {
        let mut vec = [].to_vec();
        for (idx, val) in self.vec.iter().enumerate() {
            let i = idx / self.stride;
            let j = idx % self.stride;

            if j.saturating_sub_signed(offset) <= i {
                vec.push(val.clone());
            } else {
                vec.push(T::zero());
            }
        }

        Matrix {
            vec,
            stride: self.stride,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::err::ValueError;

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

    #[test]
    fn test_triu() {
        let a = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).unwrap();
        let expected =
            Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [0, 8, 9], [0, 0, 12]]).unwrap();
        let actual = a.triu(-1);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_tril() {
        let a = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).unwrap();
        let expected =
            Matrix::try_from_iter([[0, 0, 0], [4, 0, 0], [7, 8, 0], [10, 11, 12]]).unwrap();
        let actual = a.tril(-1);

        assert_eq!(expected, actual);
    }
}
