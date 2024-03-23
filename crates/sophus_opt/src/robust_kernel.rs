/// Robust kernel functions
#[derive(Debug, Clone, Copy)]
pub struct HuberKernel {
    delta: f64,
}

/// Trait for robust kernels
pub trait IsRobustKernel {
    /// apply the kernel
    fn apply(&self, x: f64) -> f64;

    /// calculate the weight for a given residual
    fn weight(&self, residual_nrm: f64) -> f64 {
        if residual_nrm == 0.0 {
            return 0.0;
        }
        self.apply(residual_nrm).sqrt() / residual_nrm
    }
}

impl HuberKernel {
    /// create a new Huber kernel
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl IsRobustKernel for HuberKernel {
    fn apply(&self, x: f64) -> f64 {
        if x.abs() <= self.delta {
            x.powi(2)
        } else {
            2.0 * self.delta * x.abs() - self.delta.powi(2)
        }
    }
}

/// Robust kernel functions
#[derive(Debug, Clone, Copy)]
pub enum RobustKernel {
    /// Huber kernel
    Huber(HuberKernel),
}

impl IsRobustKernel for RobustKernel {
    fn apply(&self, x: f64) -> f64 {
        match self {
            RobustKernel::Huber(huber) => huber.apply(x),
        }
    }

    fn weight(&self, x: f64) -> f64 {
        match self {
            RobustKernel::Huber(huber) => huber.weight(x),
        }
    }
}
