#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot(float* a, float* b, float* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
			__syncthreads();
		}
		i /= 2;
	}
	if (cacheIndex == 0) {
		c[blockIdx.x] = cache[0];
	}
}
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);
//const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
int main() {
	float* a, * b, c=0, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;
	a = new float[N];
	b = new float[N];
	partial_c = new float[blocksPerGrid];
	cudaMalloc((void**)&dev_a, sizeof(float) * N);
	cudaMalloc((void**)&dev_b, sizeof(float) * N);
	cudaMalloc((void**)&dev_partial_c, sizeof(float) * blocksPerGrid);
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}
	cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
	
	dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);
	cudaMemcpy(partial_c, dev_partial_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
	for (int i = 0; i < blocksPerGrid;i++) {
		c += partial_c[i];
	}
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("%.6g == %.6g", c, 2 * sum_squares((float)(N - 1)));
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	delete[] a;
	delete[] b;
	delete[] partial_c;
	std::cin.get();
}