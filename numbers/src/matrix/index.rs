use std::ops::RangeBounds;

pub trait MatrixIndex {
    fn offset(&self, shape: (usize, usize)) -> usize;
    fn rows(&self, shape: (usize, usize)) -> usize;
    fn columns(&self, shape: (usize, usize)) -> usize;
}

impl<R: RangeBounds<usize>> MatrixIndex for (R, R) {
    fn offset(&self, (_, n): (usize, usize)) -> usize {
        let (row_range, column_range) = self;
        let i = match row_range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let j = match column_range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };

        i * n + j
    }

    fn rows(&self, (m, _): (usize, usize)) -> usize {
        let row_range = &self.0;

        let start = match row_range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match row_range.end_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start - 1,
            std::ops::Bound::Unbounded => m,
        };

        end.saturating_sub(start) + 1
    }

    fn columns(&self, (_, n): (usize, usize)) -> usize {
        let column_range = &self.1;

        let start = match column_range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match column_range.end_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start - 1,
            std::ops::Bound::Unbounded => n,
        };

        end.saturating_sub(start) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset() {
        let expected = 8;

        let shape: (usize, usize) = (5, 6);
        let matrix_index = (1..2, 2..3);
        let actual = matrix_index.offset(shape);

        assert_eq!(expected, actual);

        let expected = 8;

        let matrix_index = (1..=2, 2..=3);
        let actual = matrix_index.offset(shape);

        assert_eq!(expected, actual);

        let expected = 0;

        let matrix_index = (..2, ..3);
        let actual = matrix_index.offset(shape);

        assert_eq!(expected, actual);

        let expected = 8;

        let matrix_index = (1.., 2..);
        let actual = matrix_index.offset(shape);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_rows() {
        let expected = 1;

        let shape: (usize, usize) = (5, 6);
        let matrix_index = (1..2, 2..3);
        let actual = matrix_index.rows(shape);

        assert_eq!(expected, actual);

        let expected = 2;

        let matrix_index = (1..=2, 2..=3);
        let actual = matrix_index.rows(shape);

        assert_eq!(expected, actual);

        let expected = 2;

        let matrix_index = (..2, ..3);
        let actual = matrix_index.rows(shape);

        assert_eq!(expected, actual);

        let expected = 5;

        let matrix_index = (1.., 2..);
        let actual = matrix_index.rows(shape);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_columns() {
        let expected = 1;

        let shape: (usize, usize) = (5, 6);
        let matrix_index = (1..2, 2..3);
        let actual = matrix_index.columns(shape);

        assert_eq!(expected, actual);

        let expected = 2;

        let matrix_index = (1..=2, 2..=3);
        let actual = matrix_index.columns(shape);

        assert_eq!(expected, actual);

        let expected = 3;

        let matrix_index = (..2, ..3);
        let actual = matrix_index.columns(shape);

        assert_eq!(expected, actual);

        let expected = 5;

        let matrix_index = (1.., 2..);
        let actual = matrix_index.columns(shape);

        assert_eq!(expected, actual);
    }
}
