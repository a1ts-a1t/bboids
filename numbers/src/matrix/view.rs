use std::ops::Index;

use crate::{
    Matrix,
    matrix::{MatrixIndex, region::Region},
};

#[derive(PartialEq)]
pub struct View<'a, T> {
    pub(crate) buffer: &'a [T],
    pub(crate) region: Region,
}

impl<'a, T> Index<(usize, usize)> for View<'a, T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.buffer[self.region.buffer_index(i, j)]
    }
}

impl<'a, T> View<'a, T> {
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if self.region.in_bounds(i, j) {
            Some(&self[(i, j)])
        } else {
            None
        }
    }

    pub fn view<'b, I: MatrixIndex>(&'b self, index: I) -> View<'b, T> {
        let subregion = self.region.crop(index);

        View {
            buffer: self.buffer,
            region: subregion,
        }
    }

    pub fn checked_view<'b, I: MatrixIndex>(&'b self, index: I) -> Option<View<'b, T>> {
        self.region.try_crop(index).map(|subregion| View {
            buffer: self.buffer,
            region: subregion,
        })
    }
}

impl<'a, T: Clone> View<'a, T> {
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
        let region = Region::from_shape(3, 2).crop((1.., ..));

        let matrix_view = View {
            buffer: &[1, 2, 3, 4, 5, 6],
            region,
        };

        let expected = 6;
        let actual = matrix_view[(1, 1)];

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get() {
        let region = Region::from_shape(3, 2).crop((1.., ..));

        let matrix_view = View {
            buffer: &[1, 2, 3, 4, 5, 6],
            region,
        };

        let expected = &6;
        let actual = matrix_view.get(1, 1).unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get_none() {
        let region = Region::from_shape(3, 2).crop((1.., ..));

        let matrix_view = View {
            buffer: &[1, 2, 3, 4, 5, 6],
            region,
        };

        let actual = matrix_view.get(1, 2);
        assert!(actual.is_none());
    }

    #[test]
    fn test_as_matrix() {
        let region = Region::from_shape(3, 2).crop((1.., ..));

        let matrix_view = View {
            buffer: &[1, 2, 3, 4, 5, 6],
            region,
        };

        let expected = Matrix::try_from_iter([[3, 4], [5, 6]]).unwrap();
        let actual = matrix_view.as_matrix();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_view() {
        let expected = Matrix::try_from_iter([[5]]).unwrap();

        let matrix = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).unwrap();
        let view = matrix.view((0..2, 0..2));

        let actual = view.view((1..=1, 1..=1)).into_matrix();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_checked_view() {
        let matrix = Matrix::try_from_iter([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).unwrap();
        let view = matrix.view((0..2, 0..2));

        assert!(view.checked_view((0..=2, 0..=2)).is_none());
    }
}
