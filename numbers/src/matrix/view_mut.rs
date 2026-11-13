use std::ops::{Index, IndexMut};

use crate::{
    Matrix,
    matrix::{MatrixIndex, MatrixView, region::Region},
};

#[derive(PartialEq)]
pub struct MatrixViewMut<'a, T> {
    pub(crate) buffer: &'a mut [T],
    pub(crate) region: Region,
}

impl<'a, T> IndexMut<(usize, usize)> for MatrixViewMut<'a, T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.buffer[self.region.buffer_index(i, j)]
    }
}

impl<'a, T> Index<(usize, usize)> for MatrixViewMut<'a, T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.buffer[self.region.buffer_index(i, j)]
    }
}

impl<'a, T> MatrixViewMut<'a, T> {
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if self.region.in_bounds(i, j) {
            Some(&self[(i, j)])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        if self.region.in_bounds(i, j) {
            Some(IndexMut::index_mut(self, (i, j)))
        } else {
            None
        }
    }

    pub fn as_view<'b>(&'b self) -> MatrixView<'b, T> {
        MatrixView {
            buffer: self.buffer,
            region: self.region,
        }
    }

    pub fn view_mut<'b, I: MatrixIndex>(&'b mut self, index: I) -> MatrixViewMut<'b, T> {
        let region = self.region.crop(index);

        MatrixViewMut {
            buffer: self.buffer,
            region,
        }
    }

    pub fn view<'b, I: MatrixIndex>(&'b self, index: I) -> MatrixView<'b, T> {
        let region = self.region.crop(index);

        MatrixView {
            buffer: self.buffer,
            region,
        }
    }

    pub fn checked_view_mut<'b, I: MatrixIndex>(
        &'b mut self,
        index: I,
    ) -> Option<MatrixViewMut<'b, T>> {
        match self.region.try_crop(index) {
            Some(region) => Some(MatrixViewMut {
                buffer: self.buffer,
                region,
            }),
            None => None,
        }
    }

    pub fn checked_view<'b, I: MatrixIndex>(&'b self, index: I) -> Option<MatrixView<'b, T>> {
        self.region.try_crop(index).map(|region| MatrixView {
            buffer: self.buffer,
            region,
        })
    }
}

impl<'a, T: Clone> MatrixViewMut<'a, T> {
    pub fn as_matrix(&self) -> Matrix<T> {
        Matrix {
            vec: self.region.apply(self.buffer),
            stride: self.region.shape().1,
        }
    }

    pub fn into_matrix(self) -> Matrix<T> {
        self.as_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index() {
        let matrix_view = MatrixViewMut {
            buffer: &mut [1, 2, 3, 4, 5, 6],
            region: Region::from_shape(3, 2).crop((1.., ..)),
        };

        let expected = 6;
        let actual = matrix_view[(1, 1)];

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get() {
        let matrix_view = MatrixViewMut {
            buffer: &mut [1, 2, 3, 4, 5, 6],
            region: Region::from_shape(3, 2).crop((1.., ..)),
        };

        let expected = &6;
        let actual = matrix_view.get(1, 1).unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get_none() {
        let matrix_view = MatrixViewMut {
            buffer: &mut [1, 2, 3, 4, 5, 6],
            region: Region::from_shape(3, 2).crop((1.., ..)),
        };

        let actual = matrix_view.get(1, 2);
        assert!(actual.is_none());
    }

    #[test]
    fn test_get_mut() {
        let mut buffer = [1, 2, 3, 4, 5, 6];
        let mut matrix_view = MatrixViewMut {
            buffer: &mut buffer,
            region: Region::from_shape(3, 2).crop((1.., ..)),
        };

        *matrix_view.get_mut(0, 0).unwrap() = 30;

        assert_eq!(buffer, [1, 2, 30, 4, 5, 6]);
    }

    #[test]
    fn test_as_matrix() {
        let matrix_view = MatrixViewMut {
            buffer: &mut [1, 2, 3, 4, 5, 6],
            region: Region::from_shape(3, 2).crop((1.., ..)),
        };

        let expected = Matrix::try_from_iter([[3, 4], [5, 6]]).unwrap();
        let actual = matrix_view.as_matrix();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_view_mut() {
        let expected = Matrix::try_from_iter([[5]]).unwrap();

        let mut matrix = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).unwrap();
        let mut view = matrix.view_mut((0..2, 0..2));

        let actual = view.view_mut((1..=1, 1..=1)).into_matrix();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_checked_view_mut() {
        let mut matrix = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).unwrap();
        let mut view = matrix.view_mut((0..2, 0..2));

        assert!(view.checked_view_mut((0..=2, 0..=2)).is_none());
    }

    #[test]
    fn test_checked_view_mut_full_range() {
        let mut matrix = Matrix::try_from_iter([[1, 2], [3, 4]]).unwrap();
        let mut view = matrix.view_mut((.., ..));

        assert!(view.checked_view_mut((.., ..)).is_some());
    }
}
