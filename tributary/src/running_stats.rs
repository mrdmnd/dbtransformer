// ============================================================================
// Streaming stats
// ============================================================================

#[derive(Default, Clone, Copy)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn std(&self) -> f64 {
        if self.count > 1 {
            (self.m2 / (self.count - 1) as f64).sqrt().max(1e-8)
        } else {
            1.0
        }
    }

    fn has_samples(&self) -> bool {
        self.count > 0
    }
}
