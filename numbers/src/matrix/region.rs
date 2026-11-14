use crate::matrix::MatrixIndex;

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct Region {
    offset: usize,
    stride: usize,
    rows: usize,
    columns: usize,
}

impl Region {
    pub fn from_shape(m: usize, n: usize) -> Self {
        Self {
            offset: 0,
            stride: n,
            rows: m,
            columns: n,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn buffer_index(&self, i: usize, j: usize) -> usize {
        self.offset + (self.stride * i) + j
    }

    pub fn in_bounds(&self, i: usize, j: usize) -> bool {
        i < self.rows && j < self.columns
    }

    pub fn crop<I: MatrixIndex>(&self, index: I) -> Self {
        match self.try_crop(index) {
            Some(region) => region,
            None => panic!(),
        }
    }

    pub fn try_crop<I: MatrixIndex>(&self, index: I) -> Option<Self> {
        let ((row_start, row_end), (column_start, column_end)) = index.bounds(self.shape());

        if row_start > row_end
            || column_start > column_end
            || row_end > self.rows
            || column_end > self.columns
        {
            return None;
        }

        Some(Self {
            offset: self.offset + row_start * self.stride + column_start,
            stride: self.stride,
            rows: row_end - row_start,
            columns: column_end - column_start,
        })
    }

    pub fn try_crop_row(&self, i: usize) -> Option<Self> {
        self.try_crop((i..i + 1, ..))
    }

    pub fn try_crop_column(&self, j: usize) -> Option<Self> {
        self.try_crop((.., j..j + 1))
    }

    pub fn crop_row(&self, i: usize) -> Self {
        self.crop((i..i + 1, ..))
    }

    pub fn crop_column(&self, j: usize) -> Self {
        self.crop((.., j..j + 1))
    }

    pub fn apply<T: Clone>(&self, buffer: &[T]) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.rows * self.columns);

        for i in 0..self.rows {
            for j in 0..self.columns {
                vec.push(buffer[self.buffer_index(i, j)].clone());
            }
        }

        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_index() {
        let region = Region::from_shape(3, 3);
        let subregion = region.crop((0..2, 0..2));
        let nested_region = subregion.crop((1..=1, 1..=1));

        assert_eq!(nested_region.buffer_index(0, 0), 4);
    }

    #[test]
    fn test_try_crop_is_none() {
        let region = Region::from_shape(3, 3);
        assert!(region.try_crop((0..5, ..)).is_none());
    }

    #[test]
    #[should_panic]
    fn test_crop_panics() {
        let region = Region::from_shape(3, 3);

        region.crop((0..5, ..));
    }

    #[test]
    fn test_apply() {
        let region = Region::from_shape(3, 2).crop((1.., ..));
        let vec = region.apply(&[1, 2, 3, 4, 5, 6]);

        assert_eq!(vec, vec![3, 4, 5, 6]);
    }
}
