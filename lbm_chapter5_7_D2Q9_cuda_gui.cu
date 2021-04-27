#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "cpu_anim.h"
#define m 256
#define n 256
#define alpha 0.25f
#define omega (1.0 / (3.0 * alpha + 0.5))
#define twall 1.0f
#define nstep 200
struct DataBlock {
	unsigned char* output_bitmap;
	CPUAnimBitmap* bitmap;
	float* dev_rho;
	float* dev_f;
	float* dev_f_old;
	float* dev_w;
	float* dev_e;
	float frames;
};

__global__ void collision(float* f_old, float* f, float* w, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;

	f_old[y * m * 9 + x * 9 + k] = (1 - omega) * f[y * m * 9 + x * 9 + k] + omega * w[k] * rho[y * m + x];
}
__global__ void stream(float* f_old, float* f, float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;
	int xp = x - e[k * 2 + 0];
	int yp = y - e[k * 2 + 1];
	if (xp >= 0 && xp < m && yp >= 0 && yp < n) {
		f[y * m * 9 + x * 9 + k] = f_old[yp * m * 9 + xp * 9 + k];
	}

}
__global__ void boundary_tb(float* f) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	f[0 * m * 9 + x * 9 + 2] = f[1 * m * 9 + x * 9 + 2];
	f[0 * m * 9 + x * 9 + 5] = f[1 * m * 9 + x * 9 + 5];
	f[0 * m * 9 + x * 9 + 6] = f[1 * m * 9 + x * 9 + 6];

	f[(n - 1) * m * 9 + x * 9 + 7] = -f[(n - 1) * m * 9 + x * 9 + 5];
	f[(n - 1) * m * 9 + x * 9 + 4] = -f[(n - 1) * m * 9 + x * 9 + 2];
	f[(n - 1) * m * 9 + x * 9 + 8] = -f[(n - 1) * m * 9 + x * 9 + 6];
}
__global__ void boundary_lr(float* f, float* w) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	f[y * m * 9 + 0 * 9 + 1] = w[1] * twall + w[3] * twall - f[y * m * 9 + 0 * 9 + 3];
	f[y * m * 9 + 0 * 9 + 5] = w[5] * twall + w[7] * twall - f[y * m * 9 + 0 * 9 + 7];
	f[y * m * 9 + 0 * 9 + 8] = w[8] * twall + w[6] * twall - f[y * m * 9 + 0 * 9 + 6];

	f[y * m * 9 + (m - 1) * 9 + 3] = -f[y * m * 9 + (m - 1) * 9 + 1];
	f[y * m * 9 + (m - 1) * 9 + 7] = -f[y * m * 9 + (m - 1) * 9 + 5];
	f[y * m * 9 + (m - 1) * 9 + 6] = -f[y * m * 9 + (m - 1) * 9 + 8];
}
__global__ void update(float* rho, float* f) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	rho[y * m + x] = f[y * m * 9 + x * 9 + 0] + f[y * m * 9 + x * 9 + 1] + f[y * m * 9 + x * 9 + 2] + f[y * m * 9 + x * 9 + 3] + f[y * m * 9 + x * 9 + 4] + f[y * m * 9 + x * 9 + 5] + f[y * m * 9 + x * 9 + 6] + f[y * m * 9 + x * 9 + 7] + f[y * m * 9 + x * 9 + 8];
}

__global__ void rho_to_bitmap(unsigned char* bitmap, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * m;
	if (rho[offset]>=0.5 && rho[offset]<=1) {
		bitmap[offset * 4 + 0] = 255 * 2 * (rho[offset]-0.5);
		bitmap[offset * 4 + 1] = 255 * 2 * (1.0 - rho[offset]);
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}
	else if (rho[offset] < 0.5 && rho[offset] >=0) {
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 255 * 2 * rho[offset];
		bitmap[offset * 4 + 2] = 255 * 2 * (0.5 - rho[offset]);
		bitmap[offset * 4 + 3] = 255;
	}
	/*if (y <= rho[(n-1)/2*m+x] * (m - 50)) {
		bitmap[offset * 4 + 0] = 255;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}*/
}

void anim_gpu(DataBlock* d, int ticks) {
	CPUAnimBitmap* bitmap = d->bitmap;
	dim3 grid2d(m / 16, n / 16);
	dim3 grid3d(m / 16, n / 16, 9);
	dim3 threads(16, 16);
	for (int k = 0; k < nstep; k++) {
		collision << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_w, d->dev_rho);
		cudaDeviceSynchronize();
		stream << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_e);
		cudaDeviceSynchronize();
		boundary_tb << <n / 16, 16 >> > (d->dev_f);
		cudaDeviceSynchronize();
		boundary_lr << <m / 16, 16 >> > (d->dev_f, d->dev_w);
		cudaDeviceSynchronize();
		update << <grid2d, threads >> > (d->dev_rho, d->dev_f);
		cudaDeviceSynchronize();
	}
	rho_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_rho);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	++d->frames;
}

void solve(DataBlock* d) {
	dim3 grid2d(m / 16, n / 16);
	dim3 grid3d(m / 16, n / 16, 9);
	dim3 threads(16, 16);
	for (int k = 0; k < nstep; k++) {
		collision << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_w, d->dev_rho);
		cudaDeviceSynchronize();
		stream << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_e);
		cudaDeviceSynchronize();
		boundary_tb << <n / 16, 16 >> > (d->dev_f);
		cudaDeviceSynchronize();
		boundary_lr << <m / 16, 16 >> > (d->dev_f, d->dev_w);
		cudaDeviceSynchronize();
		update << <grid2d, threads >> > (d->dev_rho, d->dev_f);
		cudaDeviceSynchronize();
	}
}
void print(float* rho) {
	/*for (int y = 0; y < m; y++) {
		for (int x = 0; x < 2; x++) {
			std::cout << rho[y * m + x] << ",";
		}
		std::cout << std::endl;
	}*/
	for (int x = 0; x < m; x++) {
		std::cout <<  rho[(n - 1) / 2 * m + x] << std::endl;
	}
}
void anim_exit(DataBlock* d) {
	cudaFree(d->dev_rho);
	cudaFree(d->dev_f);
	cudaFree(d->dev_f_old);
	cudaFree(d->dev_e);
	cudaFree(d->dev_w);
}
int main() {
	float w[9] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9.
			, 1. / 36., 1. / 36., 1. / 36., 1. / 36. };
	float e[9 * 2] = { 0, 0, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1,
		-1, 1, -1, -1, 1, -1 };
	float* rho, * f, * f_old;
	DataBlock data;
	CPUAnimBitmap bitmap((int)m, (int)n, (void*)&data);
	data.bitmap = &bitmap;
	data.frames = 0;
	rho = new float[m * n];
	f = new float[m * n * 9];
	f_old = new float[m * n * 9];
	for (int x = 0; x < m; x++) {
		for (int y = 0; y < n; y++) {
			rho[x * n + y] = 0;
			for (int k = 0; k < 9; k++) {
				f[x * n * 9 + y * 9 + k] = 0;
				f_old[y * m * 9 + x * 9 + k] = 0;
			}
		}
	}
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
	cudaMalloc((void**)&data.dev_rho, sizeof(float) * m * n);
	cudaMalloc((void**)&data.dev_f, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_f_old, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_w, sizeof(float) * 9);
	cudaMalloc((void**)&data.dev_e, sizeof(float) * 9 * 2);

	cudaMemcpy(data.dev_rho, rho, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f, f, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f_old, f_old, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_w, w, sizeof(float) * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_e, e, sizeof(float) * 9 * 2, cudaMemcpyHostToDevice);

	data.bitmap->anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
	//solve(&data);
	cudaMemcpy(rho, data.dev_rho, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
	print(rho);
	
	delete[] f;
	delete[] f_old;
	delete[] rho;
}