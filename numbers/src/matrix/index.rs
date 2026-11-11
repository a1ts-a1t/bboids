use std::ops::RangeBounds;

pub trait MatrixIndex {
    fn offset(&self, shape: (usize, usize)) -> usize;
    fn rows(&self, shape: (usize, usize)) -> usize;
    fn columns(&self, shape: (usize, usize)) -> usize;
}

// TODO: tests
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
