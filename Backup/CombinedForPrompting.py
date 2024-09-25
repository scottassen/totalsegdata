# Define constants
BLOCK_SIZE = 8

# Combined CUDA kernels for Fast and Accurate methods
cuda_code = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define BLOCK_SIZE 8

__device__ float edge_length(float x1, float y1, float z1, float v1,
                             float x2, float y2, float z2, float v2, float alpha) {
    float d = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
    float dv = (v1-v2) / alpha;
    d += dv*dv;
    return sqrtf(d);
}

// Fast method kernel
// Skipping this code

// Accurate method kernels

// More method kernels for geodesic distance calculation

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
# More kernels
geodesic_filter_kernel_accurate = mod.get_function("geodesic_filter_kernel_accurate")

def Geodesic3DFilter_Vol_Fast_CUDA(V, WL, WC, WS, sigma, alpha):
    ## Skipping this code
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

    # Geodesic distance calculation method here

    # Run geodesic filter
    geodesic_filter_kernel_accurate(V_gpu, V_fil_gpu,
                                np.int32(V.shape[0]), np.int32(V.shape[1]), np.int32(V.shape[2]),
                                np.int32(WL), np.int32(WC), np.int32(WS),
                                np.float32(sigma), np.float32(alpha),
                                block=block, grid=grid)

    V_fil = np.empty_like(V)
    cuda.memcpy_dtoh(V_fil, V_fil_gpu)
    return V_fil
 
 MATLAB Code
 % Compute Intrinsic Distances within the given volume window by 
% Using the Single-Source Shortest Path Algorithm
% INPUT
% Wind(WL,WC,WS,1) : X,Y,Z,val components of the pixels in the window
% WL, WC, WS : Size of the filtering volume Window
% OUTPUT
% dis(WL,WC,WS) : Intrinsic distances from the center pixel
% Prec(WL,WC,WS,2) : Backward trajectory of geodesic

function [dis, Prec] = Volume_Dist3D_Vol(Wind,  WL, WC, WS, alpha)
Vmax = 100.00;
MI = (WL-1)/2;
MJ = (WC-1)/2;
MK = (WS-1)/2;

LOC=zeros(WC*WL*WS,'uint32');
ELM=zeros(WC*WL*WS,'uint32');
MARK=zeros(WC*WL*WS,'uint32');
SP=zeros(WC*WL*WS,'single');
dct=zeros(WC*WL*WS,3,'int32');
dis=zeros(WC*WL*WS,'single');
% Initialization of the Heap, SP
for k=-MK:MK
   for i = -MI:MI
    for j = -MJ:MJ

        icode = iencode3D(i, j,k, MI, MJ,MK);
        dct(icode,1)=i;
        dct(icode,2)=j;
        dct(icode,3)=k;
        [iw,jw,kw] = idecode3D(icode,dct);
%        fprintf('icode %d i %d j %d k %d\n',icode,iw,jw,kw);
        SP(icode) = Vmax;
        LOC(icode) = icode;
        ELM(icode) = icode;
        MARK(icode) = 0;
     end
    end
end
%fprintf('Intitialize Heap\n');
SP(1) = 0.0;
icode = iencode3D(0, 0,0, MI, MJ,MK);
elm1 = ELM(1);
iloc = LOC(icode);
LOC(icode) = 1;
ELM(1) = icode;
ELM(iloc) = elm1;
LOC(elm1) = iloc;

Prec(MI+1, MJ+1, MK+1, 1) = 0;
Prec(MI+1, MJ+1, MK+1, 2) = 0;
Prec(MI+1, MJ+1, MK+1, 3) = 0;
%printheap(SP,nrest,LOC,ELM);
%fprintf('Start shorted part find\n');
for i = 1:WL*WC*WS-1
    nrest = WL*WC*WS-i+1;
    % Find the next shortest path
    icodew = ELM(1);
    spw = SP(1);
    [iw,jw,kw] = idecode3D(icodew,dct);
%   fprintf('icodew %d iw %d jw %d kw %d\n',icodew,iw,jw,kw);
%    printheap(SP,nrest,LOC,ELM);
    % Remove the first element from the heap SP
    MARK(icodew) = 1;
    sp1 = SP(1);
    elm1 = ELM(1);
    SP(1) = SP(nrest);
    ELM(1) = ELM(nrest);
    LOC(ELM(1)) = 1;
    SP(nrest) = sp1;
    ELM(nrest) = elm1;
    LOC(elm1) = nrest;
    % CALL SHIFT DOWN
%     fprintf('Before Shift Down\n');
%      printheap(SP,nrest,LOC,ELM);
    [SP,LOC,ELM]=shiftdown(SP,1,nrest-1,LOC,ELM);
%     fprintf ('After shift down\n');
%     printheap(SP,nrest,LOC,ELM);
    % First Element is Removed
    % Update the shortest distances in the heap, SP
   for kk=-1:1 
    for ii = -1:1
        for jj = -1:1
            iz = iw + ii;
            jz = jw + jj;
            kz = kw + kk;
            if (((iz >= -MI) && (iz <= MI)) && ((jz >= -MJ) && (jz <= MJ))&&((kz >= -MK) && (kz <= MK)))
                icodez = iencode3D(iz, jz, kz, MI, MJ, MK);
              if (MARK(icodez) == 0)
                spz = SP(LOC(icodez));
                dist = edge_length(iw+MI+1, jw+MJ+1,kw+MK+1, iz+MI+1,jz+MJ+1,kz+MK+1, Wind,alpha);
                if((spw + dist)< spz)
                % The distance is less than the current distance,
                % So update the heap
                SP(LOC(icodez)) = spw + dist;
%                fprintf('Dist %f\n',dist+spw);
          
    % CALLING SHIFTUP FUNCTION

                [SP,LOC,ELM]=shiftup(SP,LOC(icodez),LOC,ELM);
 %               printheap(SP,nrest,LOC,ELM);
                Prec(iz+MI+1, jz+MJ+1, kz+MK+1, 1) = iw + MI + 1;
                Prec(iz+MI+1, jz+MJ+1, kz+MK+1, 2) = iw + MI + 1;
                Prec(iz+MI+1, jz+MJ+1, kz+MK+1, 3) = kw + MK + 1;
                
                end
              end
            end
        end
    end
   end
end
    % Output the current Distance
%    printheap(SP,121,LOC,ELM);
for k = -MK:MK
  for j = -MJ:MJ
    for i = -MI:MI
        icode = iencode3D(i, j, k, MI, MJ, MK);
        dis(i+MI+1, j+MJ+1,k+MK+1) = SP(LOC(icode));
    end
  end
end
return
end

%% Compute the distance between 2 points (iw,jw,kw) and (iz,jz,kz)

function distance = edge_length(iw, jw,kw, iz, jz,kz, Wind, alpha)
%% Skipping this code
end
%% Compute linear index of the window
%  INPUT
%  I,J  : index of the window -MI <= I <= MI, -MJ <= J <= MJ
%  MI,MJ
function icode = iencode3D(I,J,K,MI,MJ,MS)
icode = ((2*MJ+1)*(2*MI+1)*(K+MS))+(2*MJ+1)*(I+MI)+J+MJ+1;
return
end

%% decode the linear code into index of window
% INPUT
% icode : linear code
% MI,MJ
% OUTPUT
% I,J : index of window
function [i,j,k] = idecode3D(icode,dct)
i=dct(icode,1);
j=dct(icode,2);
k=dct(icode,3);
return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H,LOC,ELM]=shiftdown(H,is,ie,LOC,ELM)
% Shiftdown procedure for heap update
% Heap: H(1) is min
%
i = is;
j = 2*i;
x = H(i);
p = ELM(i);
while (j <= ie)
    if (j < ie)
        if (H(j) > H(j+1))
            j = j+1;
        end
    end
    if (x <= H(j))
        break;
    else
        H(i)=H(j);
        pp=ELM(j);
        ELM(i)=pp;
        LOC(pp)=i;
        i=j;
        j=2*i;
%        fprintf('i %d j %d\n',i,j);
    end
end
    H(i)=x;
    LOC(p)=i;
    ELM(i)=p;
return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
function [H,LOC,ELM]=shiftup(H,is,LOC,ELM)
    % Shiftup procedure for heap update
    % Heap: H(1) is min
    %
    
i=is;
j=floor(i/2);
x=H(i);
p=ELM(i);
while (j >= 1)
    if (H(j)<= x)
        break;
    end
        H(i)=H(j);
        pp=ELM(j);
        ELM(i)=pp;
        LOC(pp)=i;
        i=j;
        j=i/2;
        if(i==1)j=0;
        end
%        fprintf('**i %d j %d\n',i,j);
end
    H(i)=x;
    ELM(i)=p;
    LOC(p)=i;
return
end
function printheap(H,ie,LOC,ELM)
for i=1:ie
    fprintf('i = %d H= %f LOC = %d ELM= %d\n',i,H(i),LOC(ELM(i)),ELM(i));
end
return
end