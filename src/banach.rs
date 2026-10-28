pub trait Element: Sized + Clone {
    type ScalarElement: Clone; // should be an element of a field,,,,,, how the hell do you define field
    type NormElement: Clone + Ord; // should be a field that's ordered

    fn add(&self, other: Self) -> Self;

    fn add_ref(&self, other: &Self) -> Self {
        self.add(other.clone())
    }

    fn mul(&self, scalar: Self::ScalarElement) -> Self;

    fn mul_ref(&self, scalar: &Self::ScalarElement) -> Self {
        self.mul(scalar.clone())
    }

    fn inv(&self) -> Self;

    fn norm(&self) -> Self::NormElement;

    fn sub(&self, other: Self) -> Self {
        self.add(other.inv())
    }

    fn sub_ref(&self, other: &Self) -> Self {
        self.sub(other.clone())
    }

    fn distance(&self, other: Self) -> Self::NormElement {
        self.sub(other).norm()
    }

    fn distance_ref(&self, other: &Self) -> Self::NormElement {
        self.distance(other.clone())
    }
}

pub trait Space {
    type Element: Element;

    fn mean<I: Iterator<Item = Self::Element>>(&self, elements: I) -> Self::Element;

    fn zero(&self) -> Self::Element;
}
