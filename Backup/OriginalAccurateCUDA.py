# Accurate CUDA kernels
accurate_cuda_code = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define BLOCK_SIZE 8
#define NARROW_BAND_SIZE 64
#define FAR 1.0e6f

__device__ float edge_length(float x1, float y1, float z1, float v1,
                             float x2, float y2, float z2, float v2, float alpha) {
    float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
    float dv = (v1-v2) / alpha;
    d += dv*dv;
    return sqrtf(d);
}

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

__global__ void fast_marching_step(float* dist, int width, int height, int depth, float* V, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        update_distance(dist, x, y, z, width, height, depth, V, alpha);
    }
}

__global__ void geodesic_filter_kernel(float* V, float* V_fil, float* dist, int width, int height, int depth, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int k = -NARROW_BAND_SIZE/2; k <= NARROW_BAND_SIZE/2; k++) {
            for (int i = -NARROW_BAND_SIZE/2; i <= NARROW_BAND_SIZE/2; i++) {
                for (int j = -NARROW_BAND_SIZE/2; j <= NARROW_BAND_SIZE/2; j++) {
                    int nx = x + i, ny = y + j, nz = z + k;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                        float d = dist[(nz*height + ny)*width + nx] - dist[(z*height + y)*width + x];
                        float weight = expf(-(d*d) / (2*sigma*sigma));
                        sum += weight * V[(nz*height + ny)*width + nx];
                        weight_sum += weight;
                    }
                }
            }
        }
        
        V_fil[(z*height + y)*width + x] = sum / weight_sum;
    }
}
"""

# Compile the Accurate CUDA kernels
mod = SourceModule(accurate_cuda_code)
init_distance_field_kernel = mod.get_function("init_distance_field")
fast_marching_step_kernel = mod.get_function("fast_marching_step")
geodesic_filter_kernel = mod.get_function("geodesic_filter_kernel")

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
    for _ in range(NARROW_BAND_SIZE):
        fast_marching_step_kernel(dist_gpu, np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                                  V_gpu, np.float32(alpha),
                                  block=block, grid=grid)

    # Run geodesic filter
    geodesic_filter_kernel(V_gpu, V_fil_gpu, dist_gpu,
                           np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                           np.float32(sigma),
                           block=block, grid=grid)

    V_fil = np.empty_like(V)
    cuda.memcpy_dtoh(V_fil, V_fil_gpu)
    return V_fil

def GeoFilter3D_NRRD_Accurate_CUDA(input_file, WL, WC, WS, sigma, alpha):
    # Read input file
    image = sitk.ReadImage(input_file)
    V = sitk.GetArrayFromImage(image)
    
    # Perform volume filtering
    V_fil = Geodesic3DFilter_Vol_Accurate_CUDA(V, WL, WC, WS, sigma, alpha)
    
    # Save Filtered Volume
    output_file = f"{os.path.splitext(input_file)[0]}geo{sigma}_Accurate_CUDA.nrrd"
    output_image = sitk.GetImageFromArray(V_fil)
    output_image.CopyInformation(image)
    sitk.WriteImage(output_image, output_file)
    
    return V_fil