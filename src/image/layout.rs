use super::pixel::{ScalarTrait, P};

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

pub trait ImageSizeTrait {
    fn width(&self) -> usize {
        self.size().width
    }

    fn height(&self) -> usize {
        self.size().height
    }

    fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    fn size(&self) -> ImageSize;
}

impl ImageSizeTrait for ImageSize {
    fn size(&self) -> ImageSize {
        *self
    }
}

impl ImageSize {
    pub fn new(width: usize, height: usize) -> Self {
        ImageSize::from_width_and_height(width, height)
    }

    pub fn from_width_and_height(width: usize, height: usize) -> Self {
        ImageSize { width, height }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct ImageLayout {
    pub size: ImageSize,
    pub stride: usize,
}

impl ImageLayout {
    pub fn from_width_and_height<T>(width: usize, height: usize) -> Self {
        ImageLayout {
            size: ImageSize { width, height },
            stride: width,
        }
    }
}

pub trait ImageLayoutTrait: ImageSizeTrait {
    fn stride(&self) -> usize {
        self.layout().stride
    }

    fn padded_area(&self) -> usize {
        self.layout().stride() * self.layout().height()
    }

    fn num_bytes_of_padded_area<const NUM: usize, Scalar: ScalarTrait + 'static>(&self) -> usize {
        self.padded_area() * std::mem::size_of::<P<NUM, Scalar>>()
    }

    fn layout(&self) -> ImageLayout;
}

impl ImageSizeTrait for ImageLayout {
    fn size(&self) -> ImageSize {
        self.size
    }
}

impl ImageLayoutTrait for ImageLayout {
    fn layout(&self) -> ImageLayout {
        *self
    }
}
