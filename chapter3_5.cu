#include <iostream>
#include <cuda_runtime.h>

int main(void) {
	cudaDeviceProp prop;
	int dev;
	cudaGetDevice(&dev);
	printf("ID of current CUDA device: %d\n", dev);
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 8;
	prop.minor = 6;
	cudaChooseDevice(&dev, &prop);
	printf("ID of CUDA device closest to revision 8.6: %d\n", dev);
	cudaSetDevice(dev);
}