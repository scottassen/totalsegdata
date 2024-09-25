# Define constants
BLOCK_SIZE = 8
BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 32
BLOCK_SIZE_Z = 32

# Combined CUDA kernels for Fast and Accurate methods
cuda_code = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define BLOCK_SIZE 8
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define BLOCK_SIZE_Z 32
#define NARROW_BAND_SIZE 64
#define FAR 1.0e6f

__device__ float edge_length(float x1, float y1, float z1, float v1,
                             float x2, float y2, float z2, float v2, float alpha) {
    float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
    float dv = (v1-v2) / alpha;
    d += dv*dv;
    return sqrtf(d);
}

__global__ void geodesic_filter_kernel_fast(float* V, float* V_fil, int lines, int columns, int depth,
                                            int WL, int WC, int WS, float sigma, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx < lines && idy < columns && idz < depth) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        int MI = WL/2, MJ = WC/2, MK = WS/2;
        float center_val = V[idx*columns*depth + idy*depth + idz];
        
        for (int k = -MK; k <= MK; k++) {
            for (int i = -MI; i <= MI; i++) {
                for (int j = -MJ; j <= MJ; j++) {
                    if (idx+i >= 0 && idx+i < lines && idy+j >= 0 && idy+j < columns && idz+k >= 0 && idz+k < depth) {
                        float neighbor_val = V[(idx+i)*columns*depth + (idy+j)*depth + (idz+k)];
                        float dist = edge_length(idx, idy, idz, center_val, idx+i, idy+j, idz+k, neighbor_val, alpha);
                        float weight = expf(-(dist*dist) / (2*sigma*sigma));
                        sum += weight * neighbor_val;
                        weight_sum += weight;
                    }
                }
            }
        }
        V_fil[idx*columns*depth + idy*depth + idz] = sum / weight_sum;
    }
}

__global__ void init_distance_field(float* dist, float* V, int width, int height, int depth, int cx, int cy, int cz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int idx = (z*height + y)*width + x;
        dist[idx] = (x == cx && y == cy && z == cz) ? 0.0f : FAR;
    }
}

__global__ void update_distances_x(float* dist, float* V, int width, int height, int depth, float alpha, int dx) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (y < height && z < depth) {
        for (int x = (dx > 0) ? 0 : width - 1; x >= 0 && x < width; x += dx) {
            int idx = (z*height + y)*width + x;
            float current_dist = dist[idx];
            float current_val = V[idx];
            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    for (int k = -1; k <= 1; k++) {
                        int nx = x + i, ny = y + j, nz = z + k;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                            int nidx = (nz*height + ny)*width + nx;
                            float neighbor_dist = dist[nidx];
                            float neighbor_val = V[nidx];
                            float el = edge_length(x, y, z, current_val, nx, ny, nz, neighbor_val, alpha);
                            float new_dist = neighbor_dist + el;
                            if (new_dist < current_dist) {
                                current_dist = new_dist;
                            }
                        }
                    }
                }
            }
            
            dist[idx] = current_dist;
        }
    }
}

__global__ void update_distances_y(float* dist, float* V, int width, int height, int depth, float alpha, int dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && z < depth) {
        for (int y = (dy > 0) ? 0 : height - 1; y >= 0 && y < height; y += dy) {
            int idx = (z*height + y)*width + x;
            float current_dist = dist[idx];
            float current_val = V[idx];
            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    for (int k = -1; k <= 1; k++) {
                        int nx = x + i, ny = y + j, nz = z + k;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                            int nidx = (nz*height + ny)*width + nx;
                            float neighbor_dist = dist[nidx];
                            float neighbor_val = V[nidx];
                            float el = edge_length(x, y, z, current_val, nx, ny, nz, neighbor_val, alpha);
                            float new_dist = neighbor_dist + el;
                            if (new_dist < current_dist) {
                                current_dist = new_dist;
                            }
                        }
                    }
                }
            }
            
            dist[idx] = current_dist;
        }
    }
}

__global__ void update_distances_z(float* dist, float* V, int width, int height, int depth, float alpha, int dz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int z = (dz > 0) ? 0 : depth - 1; z >= 0 && z < depth; z += dz) {
            int idx = (z*height + y)*width + x;
            float current_dist = dist[idx];
            float current_val = V[idx];
            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    for (int k = -1; k <= 1; k++) {
                        int nx = x + i, ny = y + j, nz = z + k;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                            int nidx = (nz*height + ny)*width + nx;
                            float neighbor_dist = dist[nidx];
                            float neighbor_val = V[nidx];
                            float el = edge_length(x, y, z, current_val, nx, ny, nz, neighbor_val, alpha);
                            float new_dist = neighbor_dist + el;
                            if (new_dist < current_dist) {
                                current_dist = new_dist;
                            }
                        }
                    }
                }
            }
            
            dist[idx] = current_dist;
        }
    }
}

__global__ void geodesic_filter_kernel_accurate(float* V, float* V_fil, float* dist, int width, int height, int depth,
                                                int WL, int WC, int WS, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        int idx = (z*height + y)*width + x;
        float center_dist = dist[idx];
        int MI = WL/2, MJ = WC/2, MK = WS/2;
        
        for (int k = -MK; k <= MK; k++) {
            for (int i = -MI; i <= MI; i++) {
                for (int j = -MJ; j <= MJ; j++) {
                    int nx = x + i, ny = y + j, nz = z + k;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                        int nidx = (nz*height + ny)*width + nx;
                        float d = dist[nidx] - center_dist;
                        float weight = expf(-(d*d) / (2*sigma*sigma));
                        sum += weight * V[nidx];
                        weight_sum += weight;
                    }
                }
            }
        }
        
        V_fil[idx] = sum / weight_sum;
    }
}
"""

# Compile the combined CUDA kernels
mod = SourceModule(cuda_code)
geodesic_filter_kernel_fast = mod.get_function("geodesic_filter_kernel_fast")
init_distance_field_kernel = mod.get_function("init_distance_field")
update_distances_x_kernel = mod.get_function("update_distances_x")
update_distances_y_kernel = mod.get_function("update_distances_y")
update_distances_z_kernel = mod.get_function("update_distances_z")
geodesic_filter_kernel_accurate = mod.get_function("geodesic_filter_kernel_accurate")

def Geodesic3DFilter_Vol_Fast_CUDA(V, WL, WC, WS, sigma, alpha):
    V = V.astype(np.float32)
    V_gpu = cuda.mem_alloc(V.nbytes)
    V_fil_gpu = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(V_gpu, V)

    block = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    grid = ((V.shape[0] + block[0] - 1) // block[0],
            (V.shape[1] + block[1] - 1) // block[1],
            (V.shape[2] + block[2] - 1) // block[2])

    geodesic_filter_kernel_fast(V_gpu, V_fil_gpu,
                                np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                                np.int32(WL), np.int32(WC), np.int32(WS),
                                np.float32(sigma), np.float32(alpha),
                                block=block, grid=grid)

    V_fil = np.empty_like(V)
    cuda.memcpy_dtoh(V_fil, V_fil_gpu)
    return V_fil

def Geodesic3DFilter_Vol_Accurate_CUDA(V, WL, WC, WS, sigma, alpha):
    V = V.astype(np.float32)
    V_gpu = cuda.mem_alloc(V.nbytes)
    V_fil_gpu = cuda.mem_alloc(V.nbytes)
    dist_gpu = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(V_gpu, V)

    # Initialize distance field
    block = (8, 8, 8)
    grid = ((V.shape[0] + block[0] - 1) // block[0],
            (V.shape[1] + block[1] - 1) // block[1],
            (V.shape[2] + block[2] - 1) // block[2])
    cx, cy, cz = V.shape[0] // 2, V.shape[1] // 2, V.shape[2] // 2
    init_distance_field_kernel(dist_gpu, V_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                               np.int32(cx), np.int32(cy), np.int32(cz),
                               block=block, grid=grid)

    print("X kernel:", get_kernel_resource_usage(update_distances_x_kernel))
    print("Y kernel:", get_kernel_resource_usage(update_distances_y_kernel))
    print("Z kernel:", get_kernel_resource_usage(update_distances_z_kernel))

    optimal_size_x = optimize_block_size(update_distances_x_kernel, (256, 256, 1))
    optimal_size_y = optimize_block_size(update_distances_y_kernel, (256, 1, 256))
    optimal_size_z = optimize_block_size(update_distances_z_kernel, (256, 256, 1))

    print(f"Optimal block sizes: X: {optimal_size_x}, Y: {optimal_size_y}, Z: {optimal_size_z}")

    # Perform six passes with direction-specific kernels
    # X direction
    block_x = (1, min(BLOCK_SIZE_Y, V.shape[1]), min(BLOCK_SIZE_Z, V.shape[2]))
    grid_x = (1, (V.shape[1] + block_x[1] - 1) // block_x[1], (V.shape[2] + block_x[2] - 1) // block_x[2])
    update_distances_x_kernel(dist_gpu, V_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                              np.float32(alpha), np.int32(1), block=block_x, grid=grid_x)
    update_distances_x_kernel(dist_gpu, V_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                              np.float32(alpha), np.int32(-1), block=block_x, grid=grid_x)

    # Y direction
    block_y = (min(BLOCK_SIZE_X, V.shape[0]), 1, min(BLOCK_SIZE_Z, V.shape[2]))
    grid_y = ((V.shape[0] + block_y[0] - 1) // block_y[0], 1, (V.shape[2] + block_y[2] - 1) // block_y[2])
    update_distances_y_kernel(dist_gpu, V_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                              np.float32(alpha), np.int32(1), block=block_y, grid=grid_y)
    update_distances_y_kernel(dist_gpu, V_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                              np.float32(alpha), np.int32(-1), block=block_y, grid=grid_y)

    # Z direction
    block_z = (min(BLOCK_SIZE_X, V.shape[0]), min(BLOCK_SIZE_Y, V.shape[1]), 1) 
    grid_z = ((V.shape[0] + block_z[0] - 1) // block_z[0], (V.shape[1] + block_z[1] - 1) // block_z[1], 1)
    update_distances_z_kernel(dist_gpu, V_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                              np.float32(alpha), np.int32(1), block=block_z, grid=grid_z)
    update_distances_z_kernel(dist_gpu, V_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                              np.float32(alpha), np.int32(-1), block=block_z, grid=grid_z)

    # Run geodesic filter
    block = (8, 8, 8)
    grid = ((V.shape[0] + block[0] - 1) // block[0],
            (V.shape[1] + block[1] - 1) // block[1],
            (V.shape[2] + block[2] - 1) // block[2])
    geodesic_filter_kernel_accurate(V_gpu, V_fil_gpu, dist_gpu,
                                    np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                                    np.int32(WL), np.int32(WC), np.int32(WS),
                                    np.float32(sigma),
                                    block=block, grid=grid)

    V_fil = np.empty_like(V)
    cuda.memcpy_dtoh(V_fil, V_fil_gpu)
    return V_fil