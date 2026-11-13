use std::ops::RangeBounds;

pub trait MatrixIndex {
    fn bounds(&self, shape: (usize, usize)) -> ((usize, usize), (usize, usize));

    fn offset(&self, shape: (usize, usize)) -> usize {
        let ((row_start, _), (column_start, _)) = self.bounds(shape);
        let (_, n) = shape;

        row_start * n + column_start
    }

    fn rows(&self, shape: (usize, usize)) -> usize {
        let (start, end) = self.bounds(shape).0;
        end.saturating_sub(start)
    }

    fn columns(&self, shape: (usize, usize)) -> usize {
        let (start, end) = self.bounds(shape).1;
        end.saturating_sub(start)
    }
}

impl<RowRange: RangeBounds<usize>, ColumnRange: RangeBounds<usize>> MatrixIndex
    for (RowRange, ColumnRange)
{
    fn bounds(&self, (m, n): (usize, usize)) -> ((usize, usize), (usize, usize)) {
        let row_start = match self.0.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start.saturating_add(1),
            std::ops::Bound::Unbounded => 0,
        };

        let row_end = match self.0.end_bound() {
            std::ops::Bound::Included(end) => end.saturating_add(1),
            std::ops::Bound::Excluded(end) => *end,
            std::ops::Bound::Unbounded => m,
        };

        let column_start = match self.1.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start.saturating_add(1),
            std::ops::Bound::Unbounded => 0,
        };

        let column_end = match self.1.end_bound() {
            std::ops::Bound::Included(end) => end.saturating_add(1),
            std::ops::Bound::Excluded(end) => *end,
            std::ops::Bound::Unbounded => n,
        };

        ((row_start, row_end), (column_start, column_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset() {
        let shape: (usize, usize) = (5, 6);

        let expected = 8;
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

        let expected = 0;
        let matrix_index = (.., ..);
        let actual = matrix_index.offset(shape);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_rows() {
        let shape: (usize, usize) = (5, 6);

        let expected = 1;
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

        let expected = 4;
        let matrix_index = (1.., 2..);
        let actual = matrix_index.rows(shape);
        assert_eq!(expected, actual);

        let expected = 5;
        let matrix_index = (.., ..);
        let actual = matrix_index.rows(shape);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_columns() {
        let shape: (usize, usize) = (5, 6);

        let expected = 1;
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

        let expected = 4;
        let matrix_index = (1.., 2..);
        let actual = matrix_index.columns(shape);
        assert_eq!(expected, actual);

        let expected = 6;
        let matrix_index = (.., ..);
        let actual = matrix_index.columns(shape);
        assert_eq!(expected, actual);
    }
}
