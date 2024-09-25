# Define constants
BLOCK_SIZE = 8
NARROW_BAND_SIZE = 128

# Combined CUDA kernels for Fast and Accurate methods
cuda_code = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define BLOCK_SIZE 8
#define NARROW_BAND_SIZE 128
#define FAR 1.0e6f
#define EPSILON 1e-6f

__device__ float edge_length(float x1, float y1, float z1, float v1,
                             float x2, float y2, float z2, float v2, float alpha) {
    float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
    float dv = (v1-v2) / alpha;
    d += dv*dv;
    return sqrtf(d);
}

// Fast method kernel
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

// Accurate method kernels
__device__ void update_distance(float* dist, int x, int y, int z, int width, int height, int depth, float* V, float alpha) {
    float d[6];
    d[0] = (x > 0) ? dist[(z*height + y)*width + x-1] : FAR;
    d[1] = (x < width-1) ? dist[(z*height + y)*width + x+1] : FAR;
    d[2] = (y > 0) ? dist[(z*height + (y-1))*width + x] : FAR;
    d[3] = (y < height-1) ? dist[(z*height + (y+1))*width + x] : FAR;
    d[4] = (z > 0) ? dist[((z-1)*height + y)*width + x] : FAR;
    d[5] = (z < depth-1) ? dist[((z+1)*height + y)*width + x] : FAR;

    float a = fminf(d[0], d[1]);
    float b = fminf(d[2], d[3]);
    float c = fminf(d[4], d[5]);
    
    float t = fminf(a, fminf(b, c)) + edge_length(x, y, z, V[(z*height + y)*width + x],
                                                  x, y, z, V[(z*height + y)*width + x], alpha);
    
    dist[(z*height + y)*width + x] = fminf(dist[(z*height + y)*width + x], t);
}

__global__ void init_distance_field(float* dist, int width, int height, int depth, int cx, int cy, int cz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        dist[(z*height + y)*width + x] = (x == cx && y == cy && z == cz) ? 0.0f : FAR;
    }
}

__global__ void fast_marching_step(float* dist, int width, int height, int depth, float* V, float alpha, int* changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        float old_dist = dist[(z*height + y)*width + x];
        update_distance(dist, x, y, z, width, height, depth, V, alpha);
        if (fabsf(old_dist - dist[(z*height + y)*width + x]) > EPSILON) {
            atomicExch(changed, 1);
        }
    }
}

__global__ void normalize_distance_field(float* dist, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        dist[(z*height + y)*width + x] = fminf(dist[(z*height + y)*width + x], FAR - 1.0f);
    }
}

__global__ void geodesic_filter_kernel_accurate(float* V, float* V_fil, float* dist, int width, int height, int depth, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        float center_dist = dist[(z*height + y)*width + x];
        
        for (int k = -NARROW_BAND_SIZE/2; k <= NARROW_BAND_SIZE/2; k++) {
            for (int i = -NARROW_BAND_SIZE/2; i <= NARROW_BAND_SIZE/2; i++) {
                for (int j = -NARROW_BAND_SIZE/2; j <= NARROW_BAND_SIZE/2; j++) {
                    int nx = x + i, ny = y + j, nz = z + k;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                        float d = fabsf(dist[(nz*height + ny)*width + nx] - center_dist);
                        float weight = expf(-(d*d) / (2*sigma*sigma));
                        sum += weight * V[(nz*height + ny)*width + nx];
                        weight_sum += weight;
                    }
                }
            }
        }
        
        V_fil[(z*height + y)*width + x] = (weight_sum > EPSILON) ? sum / weight_sum : V[(z*height + y)*width + x];
    }
}
"""

# Compile the combined CUDA kernels
mod = SourceModule(cuda_code)
geodesic_filter_kernel_fast = mod.get_function("geodesic_filter_kernel_fast")
init_distance_field_kernel = mod.get_function("init_distance_field")
fast_marching_step_kernel = mod.get_function("fast_marching_step")
normalize_distance_field_kernel = mod.get_function("normalize_distance_field")
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

    block = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    grid = ((V.shape[0] + block[0] - 1) // block[0],
            (V.shape[1] + block[1] - 1) // block[1],
            (V.shape[2] + block[2] - 1) // block[2])

    # Initialize distance field
    cx, cy, cz = V.shape[0] // 2, V.shape[1] // 2, V.shape[2] // 2
    init_distance_field_kernel(dist_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                               np.int32(cx), np.int32(cy), np.int32(cz),
                               block=block, grid=grid)

    # Run Fast Marching Method
    changed_gpu = cuda.mem_alloc(4)  # 4 bytes for int
    max_iterations = 1000  # Set a maximum number of iterations
    for _ in range(max_iterations):
        cuda.memcpy_htod(changed_gpu, np.array([0], dtype=np.int32))
        fast_marching_step_kernel(dist_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                                  V_gpu, np.float32(alpha), changed_gpu,
                                  block=block, grid=grid)
        changed = np.array([0], dtype=np.int32)
        cuda.memcpy_dtoh(changed, changed_gpu)
        if changed[0] == 0:
            break

    # Normalize distance field
    normalize_distance_field_kernel(dist_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                                    block=block, grid=grid)

    # Run geodesic filter
    geodesic_filter_kernel_accurate(V_gpu, V_fil_gpu, dist_gpu,
                                    np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                                    np.float32(sigma),
                                    block=block, grid=grid)

    V_fil = np.empty_like(V)
    cuda.memcpy_dtoh(V_fil, V_fil_gpu)
    return V_fil

def GeoFilter3D_NRRD_Fast_CUDA(input_file, WL, WC, WS, sigma, alpha):
    image = sitk.ReadImage(input_file)
    V = sitk.GetArrayFromImage(image)
    V_fil = Geodesic3DFilter_Vol_Fast_CUDA(V, WL, WC, WS, sigma, alpha)
    output_file = f"{os.path.splitext(input_file)[0]}geo{sigma}_Fast_CUDA.nrrd"
    output_image = sitk.GetImageFromArray(V_fil)
    output_image.CopyInformation(image)
    sitk.WriteImage(output_image, output_file)
    return V_fil

def GeoFilter3D_NRRD_Accurate_CUDA(input_file, WL, WC, WS, sigma, alpha):
    image = sitk.ReadImage(input_file)
    V = sitk.GetArrayFromImage(image)
    V_fil = Geodesic3DFilter_Vol_Accurate_CUDA(V, WL, WC, WS, sigma, alpha)
    output_file = f"{os.path.splitext(input_file)[0]}geo{sigma}_Accurate_CUDA.nrrd"
    output_image = sitk.GetImageFromArray(V_fil)
    output_image.CopyInformation(image)
    sitk.WriteImage(output_image, output_file)
    return V_fil