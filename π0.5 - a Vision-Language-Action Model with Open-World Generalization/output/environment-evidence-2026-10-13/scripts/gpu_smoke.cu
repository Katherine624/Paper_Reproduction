#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
__global__ void add_one(float* x, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f; }
void check(cudaError_t e) { if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(1); } }
int main() {
  const int n=1<<20;
  std::vector<float> h(n,2.0f);
  float* d;
  check(cudaMalloc(&d,n*sizeof(float)));
  check(cudaMemcpy(d,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  add_one<<<(n+255)/256,256>>>(d,n);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  check(cudaMemcpy(h.data(),d,n*sizeof(float),cudaMemcpyDeviceToHost));
  check(cudaFree(d));
  for(float v:h) if(v!=3.0f) return 2;
  std::printf("PASS: 1048576 GPU additions verified\n");
}
