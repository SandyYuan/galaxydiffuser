import numpy as np
import multiprocessing as mp
from numba import jit, cuda
from scipy.spatial import cKDTree

@jit(nopython=True)
def triangular_kernel(r, l):
    return np.maximum(0, 1 - r/l)

def smooth_density_chunk(chunk, positions, l, grid_shape, grid_min, grid_max):
    tree = cKDTree(positions)
    densities = np.zeros(grid_shape)
    
    for i, j, k in np.ndindex(chunk.shape):
        point = chunk[i, j, k]
        neighbors = tree.query_ball_point(point, l)
        if neighbors:
            r = np.linalg.norm(positions[neighbors] - point, axis=1)
            weights = triangular_kernel(r, l)
            densities[i, j, k] = np.sum(weights)
    
    return densities

def smooth_density(positions, l, grid_shape, n_jobs=-1):
    """
    Compute smoothed densities from 3D positions using a triangular kernel.
    
    :param positions: Array of 3D positions (N x 3)
    :param l: Width of the triangular kernel
    :param grid_shape: Shape of the output grid (nx, ny, nz)
    :param n_jobs: Number of CPU cores to use (-1 for all available)
    :return: 3D grid of smoothed densities
    """
    grid_min = np.min(positions, axis=0)
    grid_max = np.max(positions, axis=0)
    
    # Create grid
    x = np.linspace(grid_min[0], grid_max[0], grid_shape[0])
    y = np.linspace(grid_min[1], grid_max[1], grid_shape[1])
    z = np.linspace(grid_min[2], grid_max[2], grid_shape[2])
    grid = np.meshgrid(x, y, z, indexing='ij')
    grid = np.stack(grid, axis=-1)
    
    # Split grid into chunks for parallel processing
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
    chunks = np.array_split(grid, n_jobs)
    
    # Use multiprocessing to parallelize the computation
    with mp.Pool(n_jobs) as pool:
        results = pool.starmap(smooth_density_chunk, 
                               [(chunk, positions, l, chunk.shape, grid_min, grid_max) for chunk in chunks])
    
    # Combine results
    densities = np.concatenate(results, axis=0)
    
    return densities

# GPU version (if CUDA is available)
@cuda.jit
def smooth_density_gpu(positions, l, grid, densities):
    i, j, k = cuda.grid(3)
    if i < grid.shape[0] and j < grid.shape[1] and k < grid.shape[2]:
        point = grid[i, j, k]
        density = 0.0
        for p in positions:
            r = np.sqrt((point[0]-p[0])**2 + (point[1]-p[1])**2 + (point[2]-p[2])**2)
            if r < l:
                density += max(0, 1 - r/l)
        densities[i, j, k] = density

def smooth_density_cuda(positions, l, grid_shape):
    """
    Compute smoothed densities using CUDA if available.
    
    :param positions: Array of 3D positions (N x 3)
    :param l: Width of the triangular kernel
    :param grid_shape: Shape of the output grid (nx, ny, nz)
    :return: 3D grid of smoothed densities
    """
    grid_min = np.min(positions, axis=0)
    grid_max = np.max(positions, axis=0)
    
    # Create grid
    x = np.linspace(grid_min[0], grid_max[0], grid_shape[0])
    y = np.linspace(grid_min[1], grid_max[1], grid_shape[1])
    z = np.linspace(grid_min[2], grid_max[2], grid_shape[2])
    grid = np.meshgrid(x, y, z, indexing='ij')
    grid = np.stack(grid, axis=-1)
    
    # Prepare CUDA kernel
    threadsperblock = (8, 8, 8)
    blockspergrid_x = (grid_shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (grid_shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (grid_shape[2] + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    # Allocate memory on GPU
    d_positions = cuda.to_device(positions)
    d_grid = cuda.to_device(grid)
    d_densities = cuda.device_array(grid_shape)
    
    # Run CUDA kernel
    smooth_density_gpu[blockspergrid, threadsperblock](d_positions, l, d_grid, d_densities)
    
    # Copy result back to CPU
    densities = d_densities.copy_to_host()
    
    return densities
