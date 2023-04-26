#[derive(Debug, Copy, Clone)]
#[repr(align(64))]
pub struct DataChunk([u8; 64]);

impl DataChunk {
    pub fn u8_ptr(&self) -> *const u8 {
        &self.0[0]
    }

    pub fn mut_u8_ptr(&mut self) -> *mut u8 {
        &mut self.0[0]
    }
}

impl Default for DataChunk {
    fn default() -> Self {
        DataChunk([0; 64])
    }
}

#[derive(Debug, Default, Clone)]
pub struct DataChunkVec {
    pub vec: Vec<DataChunk>,
}

impl DataChunkVec {
    pub fn slice<T>(&self) -> &[T] {
        if self.vec.len() == 0 {
            return &[];
        }
        let slice;
        unsafe {
            slice = std::slice::from_raw_parts(
                self.vec.as_ptr() as *const T,
                self.vec.len() * std::mem::size_of::<DataChunk>() / std::mem::size_of::<T>(),
            );
        }
        slice
    }

    pub fn mut_slice<T>(&mut self) -> &mut [T] {
        if self.vec.len() == 0 {
            return &mut [];
        }
        let slice;
        unsafe {
            slice = std::slice::from_raw_parts_mut(
                self.vec.as_mut_ptr() as *mut T,
                self.vec.len() * std::mem::size_of::<DataChunk>() / std::mem::size_of::<T>(),
            );
        }
        slice
    }
}
