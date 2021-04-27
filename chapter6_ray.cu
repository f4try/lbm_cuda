#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include "cpu_bitmap.h"
#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 20
#define DIM 1024
#define INF 2e10f
struct Sphere {
	float r, b, g;
	float radius;
	float x, y, z;
	__device__ float hit(float ox, float oy, float* n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return z- dz;
		}
		return INF;
	}
};



__global__ void kernel(Sphere* s, unsigned char* ptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);
	float r = 0, g = 0, b = 0;
	float minz = INF;
	for (int i = 0; i < SPHERES; i++) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t < minz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			minz = t;
		}
	}
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}


int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	Sphere* s;
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES);
	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	srand((unsigned int)time(0));
	for (int i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}
	cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);
	free(temp_s);
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grids, threads >> > (s, dev_bitmap);
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	cudaFree(dev_bitmap);
	cudaFree(s);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%3.2f ms", elapsedTime);
	bitmap.display_and_exit();
}