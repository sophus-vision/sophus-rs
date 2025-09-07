/// A 2D grid of cells.
#[derive(Debug, Clone)]
pub struct Grid<Cell: Clone> {
    cells: Vec<Cell>,
    size: [usize; 2],
}

impl<Cell: Clone> Grid<Cell> {
    /// Create a new grid with the given size and default cell value.
    pub fn new(size: [usize; 2], default: Cell) -> Self {
        Self {
            cells: vec![default; size[0] * size[1]],
            size,
        }
    }

    /// Get cell at given index.
    pub fn get(&self, idx: &[usize; 2]) -> &Cell {
        &self.cells[idx[0] * self.size[1] + idx[1]]
    }

    /// Get mutable cell at given index.
    pub fn get_mut(&mut self, idx: &[usize; 2]) -> &mut Cell {
        &mut self.cells[idx[0] * self.size[1] + idx[1]]
    }
}
