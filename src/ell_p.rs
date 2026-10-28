use std::iter::zip;

use ordered_float::OrderedFloat;

use crate::banach;

#[derive(Clone)]
pub struct Element {
    vec: Vec<f64>,
    p: f64,
}

impl banach::Element for Element {
    type ScalarElement = f64;
    type NormElement = OrderedFloat<f64>;

    fn add(&self, other: Self) -> Self {
        let vec: Vec<f64> = zip(self.vec.iter(), other.vec)
            .map(|(a, b)| a + b)
            .collect();

        Self { vec, p: self.p }
    }

    fn add_ref(&self, other: &Self) -> Self {
        let vec: Vec<f64> = zip(self.vec.iter(), other.vec.iter())
            .map(|(a, b)| a + b)
            .collect();

        Self { vec, p: self.p }
    }

    fn mul(&self, scalar: Self::ScalarElement) -> Self {
        let vec: Vec<f64> = self.vec.iter().map(|x| x * scalar).collect();

        Self { vec, p: self.p }
    }

    fn inv(&self) -> Self {
        self.mul(-1f64)
    }

    fn norm(&self) -> Self::NormElement {
        let norm = self
            .vec
            .iter()
            .map(|x| x.abs().powf(self.p))
            .sum::<f64>()
            .powf(1f64 / self.p);

        OrderedFloat(norm)
    }
}

pub struct Space {
    n: usize,
    p: f64,
}

impl banach::Space for Space {
    type Element = Element;

    fn mean<I: Iterator<Item = Self::Element>>(&self, _elements: I) -> Self::Element {
        todo!()
    }

    fn zero(&self) -> Self::Element {
        let vec = vec![0f64; self.n];
        Element { vec, p: self.p }
    }
}
