use std::ops::Index;

use crate::matrix::View;

// TODO: the same but for row
pub struct Column<'a, T>(pub(crate) View<'a, T>);

impl<'a, T> Index<usize> for Column<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[(index, 0)]
    }
}

impl<'a, T> Column<'a, T> {
    pub fn len(&self) -> usize {
        self.0.shape().0
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            Some(&self[index])
        }
    }

    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &T)> {
        self.0.enumerate().map(|((i, _), ele)| (i, ele))
    }
}
