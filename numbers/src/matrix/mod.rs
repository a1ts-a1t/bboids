mod arith;
mod column;
mod index;
mod region;
mod view;
mod view_mut;

pub use column::Column;
pub use index::MatrixIndex;
pub use view::View;
pub use view_mut::ViewMut;

use std::{
    iter,
    ops::{Index, IndexMut},
};

use num::{Float, One, Zero};

use crate::{
    Vector,
    arith::{Magnitude, Scalar},
    err::ValueError,
    matrix::region::Region,
};

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

impl<'a, T: Clone> From<View<'a, T>> for Matrix<T> {
    fn from(value: View<'a, T>) -> Self {
        value.into_matrix()
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

    pub fn view<'a, I: MatrixIndex>(&'a self, index: I) -> View<'a, T> {
        let (m, n) = self.shape();
        let region = Region::from_shape(m, n);

        View {
            buffer: &self.vec,
            region: region.crop(index),
        }
    }

    pub fn as_view<'a>(&'a self) -> View<'a, T> {
        self.view((.., ..))
    }

    pub fn view_mut<'a, I: MatrixIndex>(&'a mut self, index: I) -> ViewMut<'a, T> {
        let (m, n) = self.shape();
        let region = Region::from_shape(m, n).crop(index);

        ViewMut {
            buffer: &mut self.vec,
            region,
        }
    }

    pub fn as_view_mut<'a>(&'a mut self) -> ViewMut<'a, T> {
        self.view_mut((.., ..))
    }

    pub fn checked_view<'a, I: MatrixIndex>(&'a self, index: I) -> Option<View<'a, T>> {
        let (m, n) = self.shape();
        let region = Region::from_shape(m, n);

        region.try_crop(index).map(|view_region| View {
            buffer: &self.vec,
            region: view_region,
        })
    }

    pub fn checked_view_mut<'a, I: MatrixIndex>(&'a mut self, index: I) -> Option<ViewMut<'a, T>> {
        let (m, n) = self.shape();

        Region::from_shape(m, n)
            .try_crop(index)
            .map(|region| ViewMut {
                buffer: &mut self.vec,
                region,
            })
    }

    pub fn row<'a>(&'a self, i: usize) -> View<'a, T> {
        self.view((i..=i, ..))
    }

    pub fn column<'a>(&'a self, j: usize) -> Column<'a, T> {
        Column(self.view((.., j..=j)))
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

impl<T: PartialOrd> Matrix<T> {
    pub fn argmax(&self) -> Option<(usize, usize)> {
        let vec_index = self
            .vec
            .iter()
            .enumerate()
            .reduce(|acc, e| if e.1 > acc.1 { e } else { acc });

        vec_index.map(|(index, _)| (index / self.stride, index % self.stride))
    }
}

impl<T: Magnitude> Matrix<T> {
    pub fn abs(&self) -> Matrix<T::Output> {
        let vec = self.vec.iter().map(|ele| ele.abs()).collect();

        Matrix {
            vec,
            stride: self.stride,
        }
    }
}

impl<T> Matrix<T>
where
    T: Magnitude,
    T::Output: Scalar,
{
    pub fn norm_squared(&self) -> T::Output {
        self.vec
            .iter()
            .map(|val| val.abs() * val.abs())
            .fold(T::Output::zero(), |acc, val| acc + val)
    }
}

impl<T> Matrix<T>
where
    T: Magnitude,
    T::Output: Scalar + Float,
{
    pub fn norm(&self) -> T::Output {
        self.norm_squared().sqrt()
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

    #[test]
    fn test_checked_view() {
        let matrix = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).unwrap();
        let view = matrix.checked_view((.., ..)).unwrap();

        assert_eq!(view.into_matrix(), matrix);
    }

    #[test]
    fn test_checked_view_mut() {
        let mut matrix = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).unwrap();

        assert!(matrix.checked_view_mut((0..4, ..)).is_none());

        let mut view = matrix.checked_view_mut((0.., 1..)).unwrap();
        view[(0, 0)] = 0;

        assert_eq!(matrix[(0, 1)], 0);
    }
}
