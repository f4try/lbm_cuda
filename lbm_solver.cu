#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "cpu_anim.h"
#define m 512
#define n 128
#define r 10
#define cx 80
#define cy 63
#define bc_value 0.1
#define alpha 0.02
#define omega (1/(3.0 * alpha + 0.5))
#define twall 1.0f
#define nstep 30
struct DataBlock {
	unsigned char* output_bitmap;
	CPUAnimBitmap* bitmap;
	float* dev_rho;
	float* dev_vel;
	float* dev_f;
	float* dev_f_old;
	float* dev_w;
	float* dev_e;
	int* dev_mask;
	float frames;
};
__device__ float feq(int x, int y, int k,float* w,float* e,float* vel, float* rho) {
	float eu = e[k * 2 + 0] * vel[y * m * 2 + x * 2 + 0] + e[k * 2 + 1] * vel[y * m * 2 + x * 2 + 1];
	float uv = vel[y * m * 2 + x * 2 + 0] * vel[y * m * 2 + x * 2 + 0] + vel[y * m * 2 + x * 2 + 1] * vel[y * m * 2 + x * 2 + 1];
	return w[k]*rho[y*m+x]*(1.0+3.0*eu+4.5*eu*eu-1.5*uv);
}
__global__ void init(float* f_old, float* f, float* w, float* e, float* vel, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;
	f_old[y * m * 9 + x * 9 + k] = feq(x, y, k, w, e, vel, rho);
	f[y * m * 9 + x * 9 + k] = f_old[y * m * 9 + x * 9 + k];
}
__global__ void collision(float* f_old, float* f, float* w, float* rho, float* e, float* vel) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;

	f_old[y * m * 9 + x * 9 + k] = (1 - omega) * f[y * m * 9 + x * 9 + k] + omega * feq(x,y,k,w,e,vel,rho);
}
__global__ void stream(float* f_old, float* f, float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;
	if (x > 0 && x < m - 1 && y>0 && y < n-1) {
		int xp = x - e[k * 2 + 0];
		int yp = y - e[k * 2 + 1];
		f[y * m * 9 + x * 9 + k] = f_old[yp * m * 9 + xp * 9 + k];
	}
}

__global__ void update(float* rho, float* f, float* vel, float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > 0 && x < m - 1 && y>0 && y < n-1) {
		rho[y * m + x] = 0.0;
		vel[y * m * 2 + x * 2 + 0] = 0.0;
		vel[y * m * 2 + x * 2 + 1] = 0.0;
		for (int k = 0; k < 9; k++) {
			rho[y * m + x] += f[y * m * 9 + x * 9 + k];
			vel[y * m * 2 + x * 2 + 0] += e[k * 2 + 0] * f[y * m * 9 + x * 9 + k];
			vel[y * m * 2 + x * 2 + 1] += e[k * 2 + 1] * f[y * m * 9 + x * 9 + k];
		}
		vel[y * m * 2 + x * 2 + 0] /= rho[y * m + x];
		vel[y * m * 2 + x * 2 + 1] /= rho[y * m + x];
	}
}


__global__ void boundary_lr(float* f, float* w,float* e,float* vel,float* rho) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	vel[y * m * 2 + 0 * 2 + 0] = bc_value;
	vel[y * m * 2 + 0 * 2 + 1] = 0;
	rho[y * m + 0] = rho[y * m + 1];
	vel[y * m * 2 + (m - 1) * 2 + 0] = vel[y * m * 2 + (m - 2) * 2 + 0];
	vel[y * m * 2 + (m - 1) * 2 + 1] = vel[y * m * 2 + (m - 2) * 2 + 1];
	rho[y * m + (m - 1)] = rho[y * m + (m - 2)];
	
	for (int k = 0; k < 9;k++) {
		f[y * m * 9 + 0 * 9 + k] = feq(0, y, k, w, e, vel, rho) - feq(1, y, k, w, e, vel, rho) + f[y * m * 9 + 1 * 9 + k];
		f[y * m * 9 + (m - 1) * 9 + k] = feq(m - 1, y, k, w, e, vel, rho) - feq(m - 2, y, k, w, e, vel, rho) + f[y * m * 9 + (m - 2) * 9 + k];
	}
}

__global__ void boundary_tb(float* f, float* w, float* e, float* vel, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vel[(n - 1) * m * 2 + x * 2 + 0] = 0;
	vel[(n - 1) * m * 2 + x * 2 + 1] = 0;
	rho[(n - 1) * m  + x] = rho[(n - 2) * m + x];
	vel[0 * m * 2 + x * 2 + 0] = 0;
	vel[0 * m * 2 + x * 2 + 1] = 0;
	rho[0 * m + x] = rho[1 * m + x];

	for (int k = 0; k < 9; k++) {
		f[(n-1) * m * 9 + x * 9 + k] = feq(x, n - 1, k, w, e, vel, rho) - feq(x, n - 2, k, w, e, vel, rho) + f[(n - 2) * m * 9 + x * 9 + k];
		f[0 * m * 9 + x * 9 + k] = feq(x, 0, k, w, e, vel, rho) - feq(x, 1, k, w, e, vel, rho) + f[1 * m * 9 + x * 9 + k];
	}
}

__global__ void boundary_circle(float* vel, int* mask,float* rho,float* f,float* w,float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (mask[y * m + x] == 1) {
		vel[y * m * 2 + x * 2 + 0] = 0.0;
		vel[y * m * 2 + x * 2 + 1] = 0.0;
		int xnb = 0;
		int ynb = 0;
		if (x >= cx) {
			xnb = x + 1;
		}
		else {
			xnb = x - 1;
		}
		if (y >= cy) {
			ynb = y + 1;
		}
		else {
			ynb = y - 1;
		}
	/*	vel[y * m * 2 + x * 2 + 0] = vel[ynb * m * 2 + xnb * 2 + 0];
		vel[y * m * 2 + x * 2 + 1] = vel[ynb * m * 2 + xnb * 2 + 1];*/
		rho[y * m + x] = rho[ynb * m + xnb];

		for (int k = 0; k < 9; k++) {
			f[y * m * 9 + x * 9 + k] = feq(x, y, k, w, e, vel, rho) - feq(xnb, ynb, k, w, e, vel, rho) + f[ynb * m * 9 + xnb * 9 + k];
		}
	}
}

__device__ float vel_gradient(float* vel,int x,int y) {
	float grad[2];
	float vor;
	if (x == 0) {
		grad[0] = vel[y * m * 2 + (x + 1) * 2 + 0] - vel[y * m * 2 + x * 2 + 0];
	}
	else if (x == m - 1) {
		grad[0] = vel[y * m * 2 + x * 2 + 0] - vel[y * m * 2 + (x - 1) * 2 + 0];
	}
	else {
		grad[0] = (vel[y * m * 2 + (x + 1) * 2 + 1] - vel[y * m * 2 + (x - 1) * 2 + 1]) / 2;
	}
	if (y == 0) {
		grad[1] = vel[(y + 1) * m * 2 + x * 2 + 1] - vel[(y - 1) * m * 2 + x * 2 + 1];
	}
	else if (y == n - 1) {
		grad[1] = vel[y * m * 2 + x * 2 + 1] - vel[(y - 1) * m * 2 + x * 2 + 1];
	}
	else {
		grad[1] = (vel[(y + 1) * m * 2 + x * 2 + 1] - vel[(y - 1) * m * 2 + (x - 1) * 2 + 1]) / 2;
	}
	vor = grad[0] - grad[1];

	return vor;
}

__global__ void vel_to_bitmap(unsigned char* bitmap, float* vel) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * m;
	float vel_mag = sqrt(vel[y * m * 2 + x * 2 + 0] * vel[y * m * 2 + x * 2 + 0] + vel[y * m * 2 + x * 2 + 1] * vel[y * m * 2 + x * 2 + 1]);
	float mag = (vel_mag) / 0.15;
	/*float vor = vel_gradient(vel,x,y);
	float mag = (vor+0.02) / 0.02;*/
	if (mag>=0.5 && mag <=1) {
		bitmap[offset * 4 + 0] = 255 * 2 * (mag -0.5);
		bitmap[offset * 4 + 1] = 255 * 2 * (1 - mag);
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}
	else if (mag < 0.5 && mag >=0) {
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 255 * 2 * mag;
		bitmap[offset * 4 + 2] = 255 * 2 * (0.5 - mag);
		bitmap[offset * 4 + 3] = 255;
	}
}

void anim_gpu(DataBlock* d, int ticks) {
	CPUAnimBitmap* bitmap = d->bitmap;
	dim3 grid2d(m / 16, n / 16);
	dim3 grid3d(m / 16, n / 16, 9);
	dim3 threads(16, 16);
	init << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho);
	for (int k = 0; k < nstep; k++) {
		/*if (k % 10 == 0) {
			std::cout << k << std::endl;
		}*/
		collision << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_w, d->dev_rho, d->dev_e, d->dev_vel);
		cudaDeviceSynchronize();
		stream << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_e);
		cudaDeviceSynchronize();
		update << <grid2d, threads >> > (d->dev_rho, d->dev_f, d->dev_vel, d->dev_e);
		cudaDeviceSynchronize();
		boundary_lr << <n / 16, 16 >> > (d->dev_f, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho);
		cudaDeviceSynchronize();
		boundary_tb << <m / 16, 16 >> > (d->dev_f, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho);
		cudaDeviceSynchronize();
		boundary_circle << <grid2d, threads >> > (d->dev_vel, d->dev_mask, d->dev_rho, d->dev_f, d->dev_w, d->dev_e);
		cudaDeviceSynchronize();
	}
	vel_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_vel);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	++d->frames;
}

void solve(DataBlock* d) {
	dim3 grid2d(m / 16, n / 16);
	dim3 grid3d(m / 16, n / 16, 9);
	dim3 threads(16, 16);
	init << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho);
	for (int k = 0; k < nstep; k++) {
		/*if (k % 10 == 0) {
			std::cout << k << std::endl;
		}*/
		collision << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_w, d->dev_rho,d->dev_e,d->dev_vel);
		cudaDeviceSynchronize();
		stream << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_e);
		cudaDeviceSynchronize();
		update << <grid2d, threads >> > (d->dev_rho, d->dev_f, d->dev_vel, d->dev_e);
		cudaDeviceSynchronize();
		boundary_lr << <n / 16, 16 >> > (d->dev_f, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho);
		cudaDeviceSynchronize();
		boundary_tb << <m / 16, 16 >> > (d->dev_f, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho);
		cudaDeviceSynchronize();
		boundary_circle << <grid2d, threads >> > (d->dev_vel, d->dev_mask, d->dev_rho, d->dev_f, d->dev_w, d->dev_e);
		cudaDeviceSynchronize();
	}
}
void print(float* f) {
	for (int x = 1; x < 2; x++) {
		for (int y = 0; y < n ; y++) {
			/*std::cout << vel[y * m*2 + x * 2+0] << "," << vel[y * m * 2 + x * 2 + 1] << std::endl;*/
			for (int k = 0; k < 9; k++) {
				std::cout << f[y * m * 9 + x * 9 + k] << ",";
			}
			std::cout << std::endl;
		}
	}

	/*for (int x = 0; x < m; x++) {
		std::cout <<  rho[(n - 1) / 2 * m + x] << std::endl;
	}*/
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
	float* rho, * f, * f_old, * vel;
	int* mask;
	DataBlock data;
	CPUAnimBitmap bitmap((int)m, (int)n, (void*)&data);
	data.bitmap = &bitmap;
	data.frames = 0;
	rho = new float[m * n];
	f = new float[m * n * 9];
	f_old = new float[m * n * 9];
	mask = new int[m * n];
	vel = new float[m * n*2];
	for (int x = 0; x < m; x++) {
		for (int y = 0; y < n; y++) {
			if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
				mask[y * m + x] = 1;
			}
			else {
				mask[y * m + x] = 0;
			}
			vel[y * m * 2 + x * 2 + 0] = 0;
			vel[y * m * 2 + x * 2 + 1] = 0;
			rho[y * m + x] = 1;
			for (int k = 0; k < 9; k++) {
				f[y * m * 9 + x * 9 + k] = 0;
				f_old[y * m * 9 + x * 9 + k] = 0;
			}
		}
	}
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
	cudaMalloc((void**)&data.dev_rho, sizeof(float) * m * n);
	cudaMalloc((void**)&data.dev_vel, sizeof(float) * m * n*2);
	cudaMalloc((void**)&data.dev_f, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_f_old, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_w, sizeof(float) * 9);
	cudaMalloc((void**)&data.dev_e, sizeof(float) * 9 * 2);
	cudaMalloc((void**)&data.dev_mask, sizeof(int) * m * n);

	cudaMemcpy(data.dev_rho, rho, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_vel, vel, sizeof(float) * m * n*2, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f, f, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f_old, f_old, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_w, w, sizeof(float) * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_e, e, sizeof(float) * 9 * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_mask, mask, sizeof(int) * m * n, cudaMemcpyHostToDevice);

	data.bitmap->anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
	/*solve(&data);
	cudaMemcpy(f, data.dev_f, sizeof(float) * m * n*9, cudaMemcpyDeviceToHost);
	print(f);*/
	
	delete[] f;
	delete[] f_old;
	delete[] rho;
	delete[] vel;
}