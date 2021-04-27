#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int* c) {
	*c = a + b;
}
int main() {
	int c;
	int* dev_c;
	cudaMalloc((int**)&dev_c, sizeof(int));
	add << <1, 1 >> > (2, 7, dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << c << std::endl;
	cudaFree(dev_c);
	return 0;
}