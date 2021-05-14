#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <math.h> 
#include <string> 
#include "cpu_anim.h"
#include "gif.h"
#define m 1024
#define n 256
#define bc_value 0.1
#define react_value 0.01
#define alpha 0.10
#define omega (1/(3.0 * alpha + 0.5))
#define twall 1.0f
#define nstep 25
#define PI 3.14159265358979323846 
#define PART 1 //left 0 or right 1
#define SCALE 0.01
#define ENABLEDISPLAY true
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
	if (y > n / 3) {
		vel[y * m * 2 + 0 * 2 + 0] = bc_value;
		vel[y * m * 2 + 0 * 2 + 1] = 0;
		rho[y * m + 0] = rho[y * m + 1];
	}
	else {
		vel[y * m * 2 + 0 * 2 + 0] = 0;
		vel[y * m * 2 + 0 * 2 + 1] = 0;
		rho[y * m + 0] = rho[y * m + 1];
	}
	

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
	vel[0 * m * 2 + x * 2 + 1] = react_value;
	rho[0 * m + x] = rho[1 * m + x];

	for (int k = 0; k < 9; k++) {
		f[(n-1) * m * 9 + x * 9 + k] = feq(x, n - 1, k, w, e, vel, rho) - feq(x, n - 2, k, w, e, vel, rho) + f[(n - 2) * m * 9 + x * 9 + k];
		f[0 * m * 9 + x * 9 + k] = feq(x, 0, k, w, e, vel, rho) - feq(x, 1, k, w, e, vel, rho) + f[1 * m * 9 + x * 9 + k];
	}
}

__global__ void boundary_block(float* vel, int* mask,float* rho,float* f,float* w,float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (mask[y * m + x] !=0&&x>0&& x<m-1 &&y>0&&y<n-1) {
		vel[y * m * 2 + x * 2 + 0] = 0.0;
		vel[y * m * 2 + x * 2 + 1] = 0.0;
		//int xnb = x;
		//int ynb = y;
		//if (x>= m/2) {
		//	if (x < m - 1) {
		//		xnb = x + 1;
		//	}
		//	else {
		//		xnb = 0;
		//		//xnb = x;
		//	}
		//}
		//else {
		//	if (x >=0) {
		//		xnb = x - 1;
		//	}
		//	else {
		//		xnb = m-1;
		//		//xnb = x;
		//	}
		//}
		//if (y>=n/2) {
		//	if (y < n - 1) {
		//		ynb = y + 1;
		//	}
		//	else {
		//		ynb = 0;
		//		//ynb = y;
		//	}
		//}
		//else {
		//	if (y >0) {
		//		ynb = y - 1;
		//	}
		//	else {
		//		ynb = n-1;
		//		//ynb = y;
		//	}
		//}
		/*vel[y * m * 2 + x * 2 + 0] = vel[ynb * m * 2 + xnb * 2 + 0];
		vel[y * m * 2 + x * 2 + 1] = vel[ynb * m * 2 + xnb * 2 + 1];
		rho[y * m + x] = rho[ynb * m + xnb];*/

		/*vel[y * m * 2 + x * 2 + 0] = 0;
		vel[y * m * 2 + x * 2 + 1] = -0.05;
		rho[y * m + x] = rho[ynb * m + xnb];*/

		int vecx = 0;
		int vecy = 0;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				if (mask[(y + i) * m + x + j] == 0) {
					vecx += j;
					vecy += i;
				}
			}
		}
		int xnb = x + vecx;
		int ynb = y + vecy;
		if (react_value < 0) {
			vel[y * m * 2 + x * 2 + 0] = (abs(vel[y * m * 2 + x * 2 + 0])+0.001)*vecx * react_value;
			vel[y * m * 2 + x * 2 + 1] = (abs(vel[y * m * 2 + x * 2 + 1]) + 0.001)*vecy * react_value;
			//vel[y * m * 2 + x * 2 + 0] = vecx * react_value;
			//vel[y * m * 2 + x * 2 + 1] = vecy * react_value;
		}
		else {
			vel[y * m * 2 + x * 2 + 0] = (abs(vel[y * m * 2 + x * 2 + 0]) + 0.001) * vecx * react_value;
			vel[y * m * 2 + x * 2 + 1] = (abs(vel[y * m * 2 + x * 2 + 1]) + 0.001) * vecy * react_value;
			//vel[y * m * 2 + x * 2 + 0] = vecx * react_value;
			//vel[y * m * 2 + x * 2 + 1] = vecy * react_value;
		}
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
	float mag = (vel_mag) / SCALE;
	/*float vor = vel_gradient(vel,x,y);
	float mag = (vor+0.02) / 0.02;*/
	mag = min(mag, 1.0);
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

__global__ void mask_to_bitmap(unsigned char* bitmap, int* mask) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * m;
	
	switch (mask[offset]) {
	case 0:
		bitmap[offset * 4 + 0] = 255;
		bitmap[offset * 4 + 1] = 255;
		bitmap[offset * 4 + 2] = 255;
		bitmap[offset * 4 + 3] = 255;
		break;
	case 1:
		bitmap[offset * 4 + 0] = 255;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
		break;
	/*case 2:
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 255;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
		break;
	case 3:
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 255;
		bitmap[offset * 4 + 3] = 255;
		break;
	case 4:
		bitmap[offset * 4 + 0] = 255;
		bitmap[offset * 4 + 1] = 255;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
		break;*/
	default:
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}
}

void anim_gpu(DataBlock* d, int ticks) {
	CPUAnimBitmap* bitmap = d->bitmap;
	dim3 grid2d(m / 16, n / 16);
	dim3 grid3d(m / 16, n / 16, 9);
	//dim3 grid_fiber(m / 16, n / 16, fn);
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
		boundary_block << <grid2d, threads >> > (d->dev_vel, d->dev_mask, d->dev_rho, d->dev_f, d->dev_w, d->dev_e);
		cudaDeviceSynchronize();
	}
	vel_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_vel);
	//mask_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_mask);
	if (ENABLEDISPLAY) {
		cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	}
	++d->frames;
}

void solve(DataBlock* d) {
	dim3 grid2d(m / 16, n / 16);
	dim3 grid3d(m / 16, n / 16, 9);
	//dim3 grid_fiber(m / 16, n / 16, fn);
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
		boundary_block << <grid2d, threads >> > (d->dev_vel, d->dev_mask, d->dev_rho, d->dev_f, d->dev_w, d->dev_e);
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
using namespace std;
void imread(string filename,int* mask) {
	ifstream imgfile(filename);
	if (!imgfile.is_open())
	{
		cout << "can not open this file" << endl;
		return;
	}
	for (int y = 0; y < n; y++)
	{
		for (int x = 0; x < m; x++)
		{
			float buf;
			imgfile >> buf;
			mask[y*m+x] = (int)buf;
		}
	}
}
int main() {
	float w[9] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9.
			, 1. / 36., 1. / 36., 1. / 36., 1. / 36. };
	float e[9 * 2] = { 0, 0, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1,
		-1, 1, -1, -1, 1, -1 };
	float* rho, * f, * f_old, * vel;
	int* mask;
	//int* fibers;
	DataBlock data;
	CPUAnimBitmap bitmap((int)m, (int)n, (void*)&data);
	data.bitmap = &bitmap;
	data.frames = 0;
	rho = new float[m * n];
	f = new float[m * n * 9];
	f_old = new float[m * n * 9];
	mask = new int[m * n];
	vel = new float[m * n*2];
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < m; x++) {
			vel[y * m * 2 + x * 2 + 0] = 0;
			vel[y * m * 2 + x * 2 + 1] = 0;
			rho[y * m + x] = 1;
			for (int k = 0; k < 9; k++) {
				f[y * m * 9 + x * 9 + k] = 0;
				f_old[y * m * 9 + x * 9 + k] = 0;
			}
		}
	}
	if (PART == 0) {
		imread("xct_0414_left.txt", mask);
	}
	else {
		imread("xct_0414_right.txt", mask);
	}
	
	//imread("ketton.txt", mask);
	

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
	if (ENABLEDISPLAY) {
		data.bitmap->anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);
	}else{
		std::string filename = "./lbm_output" + std::to_string((unsigned)time(NULL)) + ".gif";
		GifWriter writer = {};
		GifBegin(&writer, filename.data(),  m, n, 20, 8, true);
		unsigned char* image;
		image = new unsigned char[m  * n * 4];
		unsigned char* image_rev;
		image_rev = new unsigned char[m  * n * 4];
		for (int frame = 0; frame < 100; ++frame)
		{
			printf("Writing frame %d...\n", frame);
			anim_gpu(&data, frame);
			cudaMemcpy(image, data.output_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
			for (int y = 0; y <  n; y++) {
				for (int x = 0; x <  m; x++) {
					image_rev[(y * m + x) * 4 + 0] = image[(( n - y - 1) * m + x) * 4 + 0];
					image_rev[(y * m + x) * 4 + 1] = image[(( n - y - 1) * m + x) * 4 + 1];
					image_rev[(y * m + x) * 4 + 2] = image[(( n - y - 1) * m + x) * 4 + 2];
					image_rev[(y * m + x) * 4 + 3] = image[(( n - y - 1) * m + x) * 4 + 3];
				}
			}

			GifWriteFrame(&writer, image_rev, m, n, 2, 8, true);
		}
		GifEnd(&writer);
	}
		/*solve(&data);
	cudaMemcpy(f, data.dev_f, sizeof(float) * m * n*9, cudaMemcpyDeviceToHost);
	print(f);*/
	
	delete[] f;
	delete[] f_old;
	delete[] rho;
	delete[] vel;
}