# Fast CUDA kernels
cuda_code = """
#include <cuda_runtime.h>

__device__ float edge_length(float x1, float y1, float z1, float v1,
                             float x2, float y2, float z2, float v2, float alpha) {
    float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
    float dv = (v1-v2) / alpha;
    d += dv*dv;
    return sqrtf(d);
}

__global__ void geodesic_filter_kernel(float* V, float* V_fil, int lines, int columns, int depth,
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
"""

# Compile the Fast CUDA kernels
mod = SourceModule(cuda_code)
geodesic_filter_kernel = mod.get_function("geodesic_filter_kernel")

def Geodesic3DFilter_Vol_Fast_CUDA(V, WL, WC, WS, sigma, alpha):
    V = V.astype(np.float32)
    V_gpu = cuda.mem_alloc(V.nbytes)
    V_fil_gpu = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(V_gpu, V)

    block = (8, 8, 8)
    grid = ((V.shape[0] + block[0] - 1) // block[0],
            (V.shape[1] + block[1] - 1) // block[1],
            (V.shape[2] + block[2] - 1) // block[2])

    geodesic_filter_kernel(V_gpu, V_fil_gpu,
                           np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                           np.int32(WL), np.int32(WC), np.int32(WS),
                           np.float32(sigma), np.float32(alpha),
                           block=block, grid=grid)

    V_fil = np.empty_like(V)
    cuda.memcpy_dtoh(V_fil, V_fil_gpu)

    return V_fil

# File processing
def GeoFilter3D_NRRD_Fast_CUDA(input_file, WL, WC, WS, sigma, alpha):
    # Read input file
    image = sitk.ReadImage(input_file)
    V = sitk.GetArrayFromImage(image)

    # Perform volume filtering
    V_fil = Geodesic3DFilter_Vol_CUDA(V, WL, WC, WS, sigma, alpha)

    # Save Filtered Volume
    output_file = f"{input_file.split('.')[0]}geo{sigma}_CUDA.nrrd"
    output_image = sitk.GetImageFromArray(V_fil)
    output_image.CopyInformation(image)
    sitk.WriteImage(output_image, output_file)

    return V_fil