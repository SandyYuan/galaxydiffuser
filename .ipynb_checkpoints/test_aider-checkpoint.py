import numpy as np
from density_smoothing import smooth_density, smooth_density_cuda

def test_smooth_density():
    # Generate random positions
    np.random.seed(42)
    positions = np.random.rand(1000, 3)
    
    # Set parameters
    l = 0.1
    grid_shape = (50, 50, 50)
    
    # Compute densities using CPU
    densities_cpu = smooth_density(positions, l, grid_shape)
    
    # Compute densities using GPU (if available)
    try:
        densities_gpu = smooth_density_cuda(positions, l, grid_shape)
        print("GPU computation successful")
        print("Max difference between CPU and GPU results:", np.max(np.abs(densities_cpu - densities_gpu)))
    except Exception as e:
        print("GPU computation failed:", str(e))
    
    print("CPU computation shape:", densities_cpu.shape)
    print("CPU computation min/max:", np.min(densities_cpu), np.max(densities_cpu))

if __name__ == "__main__":
    test_smooth_density()
