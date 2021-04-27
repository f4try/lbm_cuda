#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "cpu_anim.h"
#define m 512
#define dx 1.0f
#define alpha 0.25
#define omega (1.0 / (alpha + 0.5))
#define twall 1.0f
#define nstep 2000
struct DataBlock {
	unsigned char* output_bitmap;
	CPUAnimBitmap* bitmap;
	float* dev_rho;
	float* dev_f;
	float* dev_f_old;
	float frames;
};

__global__ void collision(float* f_old, float* f, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	for (int k = 0; k < 2; k++) {
		f_old[x * 2 + k] = (1 - omega) * f[x * 2 + k] + omega * 0.5* rho[x];
	}
	
}
__global__ void stream(float* f_old, float* f) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < m - 1) {
		f[(x+1) * 2 + 0] = f_old[x * 2 + 0];
		f[x * 2 + 1] = f_old[(x + 1) * 2 + 1];
	}
		
		
}
__global__ void boundary(float* f) {
	f[0 * 2 + 0] = twall - f[0 * 2 + 1];
	f[(m-1) * 2 + 0] = f[(m - 2) * 2 + 0];
	f[(m - 1) * 2 + 1] = f[(m - 2) * 2 + 1];
}

__global__ void update(float* rho,float* f) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	rho[x] = f[x *2 + 0] + f[x * 2 + 1];
}
__global__ void rho_to_bitmap(unsigned char* bitmap, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * m;
	if (y <= rho[x] * (m-50)) {
		bitmap[offset * 4 + 0] = 255;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}
}
void anim_gpu(DataBlock* d, int ticks) {
	CPUAnimBitmap* bitmap = d->bitmap;
	dim3 grid(m / 16, m / 16);
	dim3 threads(16, 16);
	for (int i = 0; i < nstep; i++) {
		collision << <m / 16, 16 >> > (d->dev_f_old, d->dev_f, d->dev_rho);
		cudaDeviceSynchronize();
		stream << <m / 16, 16 >> > (d->dev_f_old, d->dev_f);
		cudaDeviceSynchronize();
		boundary << <1, 1 >> > (d->dev_f);
		cudaDeviceSynchronize();
		update << <m / 16, 16 >> > (d->dev_rho, d->dev_f);
		cudaDeviceSynchronize();
	}
	rho_to_bitmap << <grid, threads >> > (d->output_bitmap, d->dev_rho);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	++d->frames;
}
void solve(DataBlock* d) {
	for (int i = 0; i < nstep; i++) {
		collision << <m / 16, 16 >> > (d->dev_f_old, d->dev_f, d->dev_rho);
		cudaDeviceSynchronize();
		stream << <m / 16, 16 >> > (d->dev_f_old, d->dev_f);
		cudaDeviceSynchronize();
		boundary << <1, 1 >> > (d->dev_f);
		cudaDeviceSynchronize();
		update << <m / 16, 16 >> > (d->dev_rho, d->dev_f);
		cudaDeviceSynchronize();
	}
}
void anim_exit(DataBlock* d) {
	cudaFree(d->dev_rho);
	cudaFree(d->dev_f);
	cudaFree(d->dev_f_old);
}

	

int main() {
	float* rho, * f, * f_old;
	DataBlock data;
	CPUAnimBitmap bitmap((int)m, (int)m, (void*)&data);
	data.bitmap = &bitmap;
	data.frames = 0;
	rho = new float[m];
	f = new float[m * 2];
	f_old = new float[m * 2];
	for (int x = 0; x < m; x++) {
		rho[x] = 0;
		for (int k = 0; k < 2; k++) {
			f[x * 2 + k] = 0;
			f_old[x * 2 + k] = 0;
		}
	}
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
	cudaMalloc((void**)&data.dev_rho, sizeof(float) * m);
	cudaMalloc((void**)&data.dev_f, sizeof(float) * m * 2);
	cudaMalloc((void**)&data.dev_f_old, sizeof(float) * m * 2);

	cudaMemcpy(data.dev_rho, rho, sizeof(float) * m, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f, f, sizeof(float) * m * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f_old, f_old, sizeof(float) * m * 2, cudaMemcpyHostToDevice);



	data.bitmap->anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
	//solve(&data);

	cudaMemcpy(rho,data.dev_rho, sizeof(float) * m , cudaMemcpyDeviceToHost);
	for (int x = 0; x < m; x++) {
		std::cout << rho[x] << ",";
	}
	std::cout << std::endl;

	delete[] f;
	delete[] f_old;
	delete[] rho;
}