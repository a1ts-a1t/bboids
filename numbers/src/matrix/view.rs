use std::ops::Index;

use crate::Matrix;

pub struct MatrixView<'a, T> {
    // resources from the underlying matrix
    pub(crate) buffer: &'a [T],
    pub(crate) stride: usize,
    // specs of the particular view
    pub(crate) offset: usize, // buffer[offset] is the first element of the view
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

impl<'a, T> Index<(usize, usize)> for MatrixView<'a, T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.buffer[self.offset + (self.stride * i) + j]
    }
}

impl<'a, T> MatrixView<'a, T> {
    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        if i >= self.rows || j >= self.columns {
            return None;
        }

        Some(&self[(i, j)])
    }
}

impl<'a, T: Clone> MatrixView<'a, T> {
    pub fn as_matrix(&self) -> Matrix<T> {
        let start_row = self.offset / self.stride;
        let start_column = self.offset % self.stride;

        let mut vec = Vec::with_capacity(self.rows * self.columns);
        for (idx, value) in self.buffer.iter().enumerate() {
            if idx < self.offset {
                continue;
            }

            let i = idx / self.stride;
            let j = idx % self.stride;

            if i >= start_row
                && i < start_row + self.rows
                && j >= start_column
                && j < start_column + self.columns
            {
                vec.push(value.clone());
            }
        }

        Matrix {
            vec,
            stride: self.columns,
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
        let matrix_view = MatrixView {
            buffer: &[1, 2, 3, 4, 5, 6],
            stride: 2,
            offset: 2,
            rows: 2,
            columns: 2,
        };

        let expected = 6;
        let actual = matrix_view[(1, 1)];

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get() {
        let matrix_view = MatrixView {
            buffer: &[1, 2, 3, 4, 5, 6],
            stride: 2,
            offset: 2,
            rows: 2,
            columns: 2,
        };

        let expected = &6;
        let actual = matrix_view.get(1, 1).unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_get_none() {
        let matrix_view = MatrixView {
            buffer: &[1, 2, 3, 4, 5, 6],
            stride: 2,
            offset: 2,
            rows: 2,
            columns: 2,
        };

        let actual = matrix_view.get(1, 2);
        assert!(actual.is_none());
    }

    #[test]
    fn test_as_matrix() {
        let matrix_view = MatrixView {
            buffer: &[1, 2, 3, 4, 5, 6],
            stride: 2,
            offset: 2,
            rows: 2,
            columns: 2,
        };

        let expected = Matrix::try_from_iter([[3, 4], [5, 6]]).unwrap();
        let actual = matrix_view.as_matrix();

        assert_eq!(expected, actual);
    }
}
