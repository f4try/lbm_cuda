#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "cpu_anim.h"
#define DIM 1024
#define SPEED 0.25f
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
struct DataBlock {
	unsigned char* output_bitmap;
	float* dev_inSrc;
	float* dev_outSrc;
	float* dev_constSrc;
	CPUAnimBitmap* bitmap;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};
__device__ unsigned char value(float n1, float n2, int hue) {
	if (hue > 360) hue -= 360;
	else if (hue < 0) hue += 360;
	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1) * (240-hue) / 60));

}
__global__ void float_to_color(unsigned char* optr, const float* outSrc) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l = outSrc[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
	float m1, m2;
	if (l <= 0.5) {
		m2 = l * (1 + s);
	}
	m1 = 2 * l - m2;
	optr[offset * 4 + 0] = value(m1, m2, h + 120);
	optr[offset * 4 + 1] = value(m1, m2, h );
	optr[offset * 4 + 2] = value(m1, m2, h - 120);
	optr[offset * 4 + 3] = 255;
}

__global__ void copy_const_kernel(float* iptr, const float* cptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	if (cptr[offset] != 0) iptr[offset] = cptr[offset];
}

__global__ void blend_kernel(float* outSrc, const float* inSrc) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	int left = offset - 1;
	int right = offset + 1;
	if (x == 0) left++;
	if (x == DIM - 1) right--;
	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0) top += DIM;
	if (y == DIM - 1) bottom -= DIM;
	outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
}

void anim_gpu(DataBlock* d,int ticks) {
	cudaEventRecord(d->start, 0);
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	CPUAnimBitmap* bitmap = d->bitmap;
	for (int i = 0; i < 90; i++) {
		copy_const_kernel << <blocks, threads >> > (d->dev_inSrc, d->dev_constSrc);
		blend_kernel << <blocks, threads >> > (d->dev_outSrc, d->dev_inSrc);
		std::swap(d->dev_inSrc, d->dev_outSrc);
	}
	float_to_color << <blocks, threads >> > (d->output_bitmap, d->dev_inSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average time per frame: %3.1f ms\n", d->totalTime / d->frames);
}
void anim_exit(DataBlock* d) {
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}

int main() {
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
	cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());
	float* temp = (float*)malloc(bitmap.image_size());
	for (int i = 0; i < DIM * DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601)) {
			temp[i] = MAX_TEMP;
		}
	}
	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;
	for (int y = 800; y < 900; y++) {
		for (int x = 400; x < 500; x++) {
			temp[x + y * DIM] = MIN_TEMP;
		}
	}
	cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(),cudaMemcpyHostToDevice);
	for (int y = 800; y < DIM; y++) {
		for (int x = 0; x < 200; x++) {
			temp[x + y * DIM] = MAX_TEMP;
		}
	}
	cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
	
}