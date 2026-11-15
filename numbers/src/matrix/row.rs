use std::ops::Index;

use crate::matrix::View;

pub struct Row<'a, T>(pub(crate) View<'a, T>);

impl<'a, T> Index<usize> for Row<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[(0, index)]
    }
}

impl<'a, T> Row<'a, T> {
    pub fn len(&self) -> usize {
        self.0.shape().1
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            Some(&self[index])
        }
    }

    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &T)> {
        self.0.enumerate().map(|((_, j), ele)| (j, ele))
    }
}
