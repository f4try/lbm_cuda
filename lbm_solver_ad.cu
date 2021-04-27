#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <time.h> 
#include <math.h> 
#include <string>
#include "cpu_anim.h"
#include "gif.h"
#define m 256
#define n 256
#define fn 15
#define fl 100
#define fw 10
#define c2 (1.0/3.0)
#define u 0.1
#define v -0.4
#define bc_value 0.1
#define alpha 0.2
#define omega (1/(3.0 * alpha + 0.5))
#define alpha_c 1.00
#define omega_c (1/(3.0 * alpha_c + 0.5))
#define twall 1.0f
#define tout 0.5f
#define nstep 20
#define PI 3.14159265358979323846 
struct DataBlock {
	unsigned char* output_bitmap;
	CPUAnimBitmap* bitmap;
	float* dev_rho;
	float* dev_vel;
	float* dev_f;
	float* dev_f_old;
	float* dev_rho_c;
	float* dev_f_c;
	float* dev_f_c_old;
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
__device__ float feq_c(int x, int y, int k, float* w, float* e, float* vel, float* rho_c) {
	float eu = e[k * 2 + 0] * vel[y * m * 2 + x * 2 + 0] + e[k * 2 + 1] * vel[y * m * 2 + x * 2 + 1];
	//float eu = e[k * 2 + 0] * u + e[k * 2 + 1] * v;
	return w[k] * rho_c[y * m + x] * (1.0 + 3.0 * eu);
}
__global__ void init(float* f_old, float* f, float* f_c_old, float* f_c, float* w, float* e, float* vel, float* rho, float* rho_c) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;
	f_old[y * m * 9 + x * 9 + k] = feq(x, y, k, w, e, vel, rho);
	f[y * m * 9 + x * 9 + k] = f_old[y * m * 9 + x * 9 + k];
	f_c_old[y * m * 9 + x * 9 + k] = feq_c(x, y, k, w, e, vel, rho_c);
	f_c[y * m * 9 + x * 9 + k] = f_c_old[y * m * 9 + x * 9 + k];
}
__global__ void collision(float* f_old, float* f, float* f_c_old, float* f_c, float* w, float* rho, float* rho_c, float* e, float* vel) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;

	f_old[y * m * 9 + x * 9 + k] = (1 - omega) * f[y * m * 9 + x * 9 + k] + omega * feq(x,y,k,w,e,vel,rho);
	f_c_old[y * m * 9 + x * 9 + k] = (1 - omega_c) * f_c[y * m * 9 + x * 9 + k] + omega_c * feq_c(x, y, k, w, e, vel, rho_c);
}
__global__ void stream(float* f_old, float* f, float* f_c_old, float* f_c, float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z;
	if (x > 0 && x < m - 1 && y>0 && y < n-1) {
		int xp = x - e[k * 2 + 0];
		int yp = y - e[k * 2 + 1];
		f[y * m * 9 + x * 9 + k] = f_old[yp * m * 9 + xp * 9 + k];
		f_c[y * m * 9 + x * 9 + k] = f_c_old[yp * m * 9 + xp * 9 + k];
	}
}

__global__ void update(float* rho, float* rho_c, float* f, float* f_c, float* vel, float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > 0 && x < m - 1 && y>0 && y < n-1) {
		rho[y * m + x] = 0.0;
		rho_c[y * m + x] = 0.0;
		vel[y * m * 2 + x * 2 + 0] = 0.0;
		vel[y * m * 2 + x * 2 + 1] = 0.0;
		for (int k = 0; k < 9; k++) {
			rho[y * m + x] += f[y * m * 9 + x * 9 + k];
			rho_c[y * m + x] += f_c[y * m * 9 + x * 9 + k];
			vel[y * m * 2 + x * 2 + 0] += e[k * 2 + 0] * f[y * m * 9 + x * 9 + k];
			vel[y * m * 2 + x * 2 + 1] += e[k * 2 + 1] * f[y * m * 9 + x * 9 + k];
		}
		vel[y * m * 2 + x * 2 + 0] /= rho[y * m + x];
		vel[y * m * 2 + x * 2 + 1] /= rho[y * m + x];
	}
}


__global__ void boundary_lr(float* f, float* f_c, float* w,float* e,float* vel,float* rho, float* rho_c) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	vel[y * m * 2 + 0 * 2 + 0] = bc_value;
	vel[y * m * 2 + 0 * 2 + 1] = 0;
	rho[y * m + 0] = rho[y * m + 1];
	rho_c[y * m + 0] = rho_c[y * m + 1];
	vel[y * m * 2 + (m - 1) * 2 + 0] = vel[y * m * 2 + (m - 2) * 2 + 0];
	vel[y * m * 2 + (m - 1) * 2 + 1] = vel[y * m * 2 + (m - 2) * 2 + 1];
	rho[y * m + (m - 1)] = rho[y * m + (m - 2)];
	rho_c[y * m + (m - 1)] = rho_c[y * m + (m - 2)];

	f_c[y * m * 9 + 0 * 9 + 1] = w[1] *twall + w[3]*twall-f_c[y * m * 9 + 0 * 9 + 3];
	f_c[y * m * 9 + 0 * 9 + 5] = w[5] * twall + w[7] * twall - f_c[y * m * 9 + 0 * 9 + 7];
	f_c[y * m * 9 + 0 * 9 + 8] = w[8] * twall + w[6] * twall - f_c[y * m * 9 + 0 * 9 + 8];/*
	f_c[y * m * 9 + (m-1) * 9 + 3] = w[1] * tout + w[3] * tout - f_c[y * m * 9 + (m - 1) * 9 + 1];
	f_c[y * m * 9 + (m - 1) * 9 + 7] = w[5] * tout + w[7] * tout - f_c[y * m * 9 + (m - 1) * 9 + 5];
	f_c[y * m * 9 + (m - 1) * 9 + 6] = w[8] * tout + w[6] * tout - f_c[y * m * 9 + (m - 1) * 9 + 8];*/
	f_c[y * m * 9 + (m - 1) * 9 + 3] = - f_c[y * m * 9 + (m - 1) * 9 + 1];
	f_c[y * m * 9 + (m - 1) * 9 + 7] = - f_c[y * m * 9 + (m - 1) * 9 + 5];
	f_c[y * m * 9 + (m - 1) * 9 + 6] = - f_c[y * m * 9 + (m - 1) * 9 + 8];
	for (int k = 0; k < 9;k++) {
		f[y * m * 9 + 0 * 9 + k] = feq(0, y, k, w, e, vel, rho) - feq(1, y, k, w, e, vel, rho) + f[y * m * 9 + 1 * 9 + k];
		f[y * m * 9 + (m - 1) * 9 + k] = feq(m - 1, y, k, w, e, vel, rho) - feq(m - 2, y, k, w, e, vel, rho) + f[y * m * 9 + (m - 2) * 9 + k];
	}
}

__global__ void boundary_tb(float* f, float* f_c, float* w, float* e, float* vel, float* rho, float* rho_c) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vel[(n - 1) * m * 2 + x * 2 + 0] = 0;
	vel[(n - 1) * m * 2 + x * 2 + 1] = 0;
	rho[(n - 1) * m  + x] = rho[(n - 2) * m + x];
	rho_c[(n - 1) * m + x] = rho_c[(n - 2) * m + x];
	vel[0 * m * 2 + x * 2 + 0] = 0;
	vel[0 * m * 2 + x * 2 + 1] = 0;
	rho[0 * m + x] = rho[1 * m + x];
	rho_c[0 * m + x] = rho_c[1 * m + x];
	/*f_c[0 * m * 9 + x * 9 + 2] = -f_c[0 * m * 9 + x * 9 + 4];
	f_c[0 * m * 9 + x * 9 + 5] = -f_c[0 * m * 9 + x * 9 + 7];
	f_c[0 * m * 9 + x * 9 + 6] = -f_c[0 * m * 9 + x * 9 + 8];

	f_c[(n-1) * m * 9 + x * 9 + 7] = -f_c[(n - 1) * m * 9 + x * 9 + 5];
	f_c[(n - 1) * m * 9 + x * 9 + 4] = -f_c[(n - 1) * m * 9 + x * 9 + 2];
	f_c[(n - 1) * m * 9 + x * 9 + 8] = -f_c[(n - 1) * m * 9 + x * 9 + 6 ];*/
	f_c[0 * m * 9 + x * 9 + 2] = f_c[1 * m * 9 + x * 9 + 2];
	f_c[0 * m * 9 + x * 9 + 5] = f_c[1 * m * 9 + x * 9 + 5];
	f_c[0 * m * 9 + x * 9 + 6] = f_c[1 * m * 9 + x * 9 + 6];

	f_c[(n - 1) * m * 9 + x * 9 + 7] = f_c[(n - 2) * m * 9 + x * 9 + 7];
	f_c[(n - 1) * m * 9 + x * 9 + 4] = f_c[(n - 2) * m * 9 + x * 9 + 4];
	f_c[(n - 1) * m * 9 + x * 9 + 8] = f_c[(n - 2) * m * 9 + x * 9 + 8];
	for (int k = 0; k < 9; k++) {
		f[(n-1) * m * 9 + x * 9 + k] = feq(x, n - 1, k, w, e, vel, rho) - feq(x, n - 2, k, w, e, vel, rho) + f[(n - 2) * m * 9 + x * 9 + k];
		f[0 * m * 9 + x * 9 + k] = feq(x, 0, k, w, e, vel, rho) - feq(x, 1, k, w, e, vel, rho) + f[1 * m * 9 + x * 9 + k];
	}
}

__global__ void boundary_fiber(float* vel, int* mask,float* rho, float* rho_c, float* f, float* f_c, float* w,float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (mask[y * m + x] !=0) {
		vel[y * m * 2 + x * 2 + 0] = 0.0;
		vel[y * m * 2 + x * 2 + 1] = 0.0;
		int xnb = 0;
		int ynb = 0;
		if (mask[y * m + x]==1|| mask[y * m + x] == 4) {
			xnb = x + 1;
			f_c[y * m * 9 + x * 9 + 1] = - f_c[y * m * 9 + x * 9 + 3];
			f_c[y * m * 9 + x * 9 + 5] = - f_c[y * m * 9 + x * 9 + 7];
			f_c[y * m * 9 + x * 9 + 8] = - f_c[y * m * 9 + x * 9 + 8];
		}
		else {
			xnb = x - 1;
			f_c[y * m * 9 + x * 9 + 3] = -f_c[y * m * 9 + x * 9 + 1];
			f_c[y * m * 9 + x * 9 + 7] = -f_c[y * m * 9 + x * 9 + 5];
			f_c[y * m * 9 + x * 9 + 6] = -f_c[y * m * 9 + x * 9 + 8];
		}
		if (mask[y * m + x] == 1 || mask[y * m + x] == 2) {
			ynb = y + 1;
			f_c[y * m * 9 + x * 9 + 7] = -f_c[y * m * 9 + x * 9 + 5];
			f_c[y * m * 9 + x * 9 + 4] = -f_c[y * m * 9 + x * 9 + 2];
			f_c[y * m * 9 + x * 9 + 8] = -f_c[y * m * 9 + x * 9 + 6];
		}
		else {
			ynb = y - 1;
			f_c[y * m * 9 + x * 9 + 2] = -f_c[y * m * 9 + x * 9 + 4];
			f_c[y * m * 9 + x * 9 + 5] = -f_c[y * m * 9 + x * 9 + 7];
			f_c[y * m * 9 + x * 9 + 6] = -f_c[y * m * 9 + x * 9 + 8];
		}
	/*	vel[y * m * 2 + x * 2 + 0] = vel[ynb * m * 2 + xnb * 2 + 0];
		vel[y * m * 2 + x * 2 + 1] = vel[ynb * m * 2 + xnb * 2 + 1];*/
		rho[y * m + x] = rho[ynb * m + xnb];
		//rho_c[y * m + x] = rho_c[ynb * m + xnb];
		//rho_c[y * m + x] = rho_c[ynb * m + xnb];
		

		/*f_c[y * m * 9 + x * 9 + 7] = -f_c[y * m * 9 + x * 9 + 5];
		f_c[y * m * 9 + x * 9 + 4] = -f_c[y * m * 9 + x * 9 + 2];
		f_c[y * m * 9 + x * 9 + 8] = -f_c[y * m * 9 + x * 9 + 6];*/
		for (int k = 0; k < 9; k++) {
			f[y * m * 9 + x * 9 + k] = feq(x, y, k, w, e, vel, rho) - feq(xnb, ynb, k, w, e, vel, rho) + f[ynb * m * 9 + xnb * 9 + k];
		}
	}
}

__global__ void boundary_block(float* vel, int* mask, float* rho, float* rho_c, float* f, float* f_c, float* w, float* e) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (mask[y * m + x] != 0) {
		vel[y * m * 2 + x * 2 + 0] = 0.0;
		vel[y * m * 2 + x * 2 + 1] = 0.0;
		int xnb = 0;
		int ynb = 0;
		if (x >= m / 2) {
			f_c[y * m * 9 + x * 9 + 1] = -f_c[y * m * 9 + x * 9 + 3];
			f_c[y * m * 9 + x * 9 + 5] = -f_c[y * m * 9 + x * 9 + 7];
			f_c[y * m * 9 + x * 9 + 8] = -f_c[y * m * 9 + x * 9 + 8];
			if (x < m - 1) {
				xnb = x + 1;
				
			}
			else {
				xnb = 0;
				//xnb = x;
			}
		}
		else {
			f_c[y * m * 9 + x * 9 + 3] = -f_c[y * m * 9 + x * 9 + 1];
			f_c[y * m * 9 + x * 9 + 7] = -f_c[y * m * 9 + x * 9 + 5];
			f_c[y * m * 9 + x * 9 + 6] = -f_c[y * m * 9 + x * 9 + 8];
			if (x >= 0) {
				xnb = x - 1;
			}
			else {
				xnb = m - 1;
				//xnb = x;
			}
		}
		if (y >= n / 2) {
			f_c[y * m * 9 + x * 9 + 7] = -f_c[y * m * 9 + x * 9 + 5];
			f_c[y * m * 9 + x * 9 + 4] = -f_c[y * m * 9 + x * 9 + 2];
			f_c[y * m * 9 + x * 9 + 8] = -f_c[y * m * 9 + x * 9 + 6];
			if (y < n - 1) {
				ynb = y + 1;
			}
			else {
				ynb = 0;
				//ynb = y;
			}
		}
		else {
			f_c[y * m * 9 + x * 9 + 2] = -f_c[y * m * 9 + x * 9 + 4];
			f_c[y * m * 9 + x * 9 + 5] = -f_c[y * m * 9 + x * 9 + 7];
			f_c[y * m * 9 + x * 9 + 6] = -f_c[y * m * 9 + x * 9 + 8];
			if (y > 0) {
				ynb = y - 1;
			}
			else {
				ynb = n - 1;
				//ynb = y;
			}
		}
		/*vel[y * m * 2 + x * 2 + 0] = vel[ynb * m * 2 + xnb * 2 + 0];
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
	int offset = x + (y+n) * m*3;
	float vel_mag = sqrt(vel[y * m * 2 + x * 2 + 0] * vel[y * m * 2 + x * 2 + 0] + vel[y * m * 2 + x * 2 + 1] * vel[y * m * 2 + x * 2 + 1]);
	float mag = (vel_mag) / (bc_value*1.5);
	mag = min(1.0, mag);
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

__global__ void rho_to_bitmap(unsigned char* bitmap, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = m+x + (y+n) * m*3;
	float mag = (rho[x+y*m]) / (twall*1.2);
	mag = min(1.0, mag);
	/*float vor = vel_gradient(vel,x,y);
	float mag = (vor+0.02) / 0.02;*/
	if (mag >= 0.5 && mag <= 1) {
		bitmap[offset * 4 + 0] = 255 * 2 * (mag - 0.5);
		bitmap[offset * 4 + 1] = 255 * 2 * (1 - mag);
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}
	else if (mag < 0.5 && mag >= 0) {
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 255 * 2 * mag;
		bitmap[offset * 4 + 2] = 255 * 2 * (0.5 - mag);
		bitmap[offset * 4 + 3] = 255;
	}
}


__global__ void mask_to_bitmap(unsigned char* bitmap, int* mask) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = 2*m+x + (y+n) * m*3;
	
	switch (mask[x + y * m]) {
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
	case 2:
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
		break;
	default:
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}
}
__global__ void mask_img_to_bitmap(unsigned char* bitmap, int* mask) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = 2 * m + x + (y + n) * m * 3;

	switch (mask[x + y * m]) {
	case 0:
		bitmap[offset * 4 + 0] = 255;
		bitmap[offset * 4 + 1] = 255;
		bitmap[offset * 4 + 2] = 255;
		bitmap[offset * 4 + 3] = 255;
		break;
	case 1:
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
		break;
	default:
		bitmap[offset * 4 + 0] = 0;
		bitmap[offset * 4 + 1] = 0;
		bitmap[offset * 4 + 2] = 0;
		bitmap[offset * 4 + 3] = 255;
	}
}
__global__ void vel_ave_to_bitmap(unsigned char* bitmap, float* vel) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0;
	for (int y = 0; y < n; y++) {
		float vel_mag = sqrt(vel[y * m * 2 + x * 2 + 0] * vel[y * m * 2 + x * 2 + 0] + vel[y * m * 2 + x * 2 + 1] * vel[y * m * 2 + x * 2 + 1]);
		sum += vel_mag;
	}
	float mag = sum / n / (bc_value * 1.5);
	mag = min(1.0, mag);
	for (int y = 0; y < n * mag; y++) {
		bitmap[y * m * 3 * 4 + x * 4 + 0] = 0;
		bitmap[y * m * 3 * 4 + x * 4 + 1] = 255;
		bitmap[y * m * 3 * 4 + x * 4 + 2] = 255;
		bitmap[y * m * 3 * 4 + x * 4 + 3] = 255;
	}
}
__global__ void rho_ave_to_bitmap(unsigned char* bitmap, float* rho) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0;
	for (int y = 0; y < n; y++) {
		sum += rho[x + y * m];
	}
	float mag = sum/n/ (twall * 1.2);
	mag = min(1.0, mag);
	for (int y = 0; y < n*mag; y++) {
		bitmap[y * m * 3 * 4 + (x + m) * 4 + 0] = 255;
		bitmap[y * m * 3 * 4 + (x + m) * 4 + 1] = 255;
		bitmap[y * m * 3 * 4 + (x + m) * 4 + 2] = 0;
		bitmap[y * m * 3 * 4 + (x + m) * 4 + 3] = 255;
	}
}
__global__ void rho_y_to_bitmap(unsigned char* bitmap, float* rho) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	float mag = rho[m*3/4 + y * m]/ (twall * 0.01);
	mag = min(1.0, mag);
	for (int i = 0; i < n * mag; i++) {
		bitmap[i * m * 3 * 4 + (y + 2*m) * 4 + 0] = 255;
		bitmap[i * m * 3 * 4 + (y + 2 * m) * 4 + 1] = 0;
		bitmap[i * m * 3 * 4 + (y + 2 * m) * 4 + 2] = 255;
		bitmap[i * m * 3 * 4 + (y + 2 * m) * 4 + 3] = 255;
	}
}
//__global__ void vel_arrow_to_bitmap(unsigned char* bitmap, int* mask,float* vel) {
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = 2 * m + x + y * m * 3;
//
//	switch (mask[x + y * m]) {
//	case 0:
//		bitmap[offset * 4 + 0] = 255;
//		bitmap[offset * 4 + 1] = 255;
//		bitmap[offset * 4 + 2] = 255;
//		bitmap[offset * 4 + 3] = 255;
//		break;
//	case 1:
//		bitmap[offset * 4 + 0] = 255;
//		bitmap[offset * 4 + 1] = 0;
//		bitmap[offset * 4 + 2] = 0;
//		bitmap[offset * 4 + 3] = 255;
//		break;
//	case 2:
//		bitmap[offset * 4 + 0] = 0;
//		bitmap[offset * 4 + 1] = 255;
//		bitmap[offset * 4 + 2] = 0;
//		bitmap[offset * 4 + 3] = 255;
//		break;
//	case 3:
//		bitmap[offset * 4 + 0] = 0;
//		bitmap[offset * 4 + 1] = 0;
//		bitmap[offset * 4 + 2] = 255;
//		bitmap[offset * 4 + 3] = 255;
//		break;
//	case 4:
//		bitmap[offset * 4 + 0] = 255;
//		bitmap[offset * 4 + 1] = 255;
//		bitmap[offset * 4 + 2] = 0;
//		bitmap[offset * 4 + 3] = 255;
//		break;
//	default:
//		bitmap[offset * 4 + 0] = 0;
//		bitmap[offset * 4 + 1] = 0;
//		bitmap[offset * 4 + 2] = 0;
//		bitmap[offset * 4 + 3] = 255;
//	}
//	if (x % 16 == 0&&y%16==0) {
//		float vel_mag = sqrt(vel[y * m * 2 + x * 2 + 0] * vel[y * m * 2 + x * 2 + 0] + vel[y * m * 2 + x * 2 + 1] * vel[y * m * 2 + x * 2 + 1]);
//		for (int i = 0; i < vel_mag*200 && i < m; i++) {
//			bitmap[(int)(2 * m + x+i* vel[y * m * 2 + x * 2 + 0] + (y+ i * vel[y * m * 2 + x * 2 + 1]) * m * 3) * 4 + 0] = 255;
//			bitmap[(int)(2 * m + x + i * vel[y * m * 2 + x * 2 + 0] + (y + i * vel[y * m * 2 + x * 2 + 1]) * m * 3) * 4 + 1] = 255;
//			bitmap[(int)(2 * m + x + i * vel[y * m * 2 + x * 2 + 0] + (y + i * vel[y * m * 2 + x * 2 + 1]) * m * 3) * 4 + 2] = 255;
//			bitmap[(int)(2 * m + x + i * vel[y * m * 2 + x * 2 + 0] + (y + i * vel[y * m * 2 + x * 2 + 1]) * m * 3) * 4 + 3] = 255;
//		}
//	}
//	
//	bitmap[offset * 4 + 0] = 0;
//	bitmap[offset * 4 + 1] = 0;
//	bitmap[offset * 4 + 2] = 0;
//	bitmap[offset * 4 + 3] = 255;
//}
void anim_gpu(DataBlock* d, int ticks) {
	CPUAnimBitmap* bitmap = d->bitmap;
	dim3 grid2d(m / 16, n / 16);
	dim3 grid3d(m / 16, n / 16, 9);
	dim3 grid_fiber(m / 16, n / 16, fn);
	dim3 threads(16, 16);
	init << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_f_c_old, d->dev_f_c, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho, d->dev_rho_c);
	for (int k = 0; k < nstep; k++) {
		/*if (k % 10 == 0) {
			std::cout << k << std::endl;
		}*/
		collision << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_f_c_old, d->dev_f_c, d->dev_w, d->dev_rho, d->dev_rho_c, d->dev_e, d->dev_vel);
		cudaDeviceSynchronize();
		stream << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_f_c_old, d->dev_f_c, d->dev_e);
		cudaDeviceSynchronize();
		update << <grid2d, threads >> > (d->dev_rho, d->dev_rho_c, d->dev_f, d->dev_f_c, d->dev_vel, d->dev_e);
		cudaDeviceSynchronize();
		boundary_lr << <n / 16, 16 >> > (d->dev_f, d->dev_f_c, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho, d->dev_rho_c);
		cudaDeviceSynchronize();
		boundary_tb << <m / 16, 16 >> > (d->dev_f, d->dev_f_c, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho, d->dev_rho_c);
		cudaDeviceSynchronize();
		/*boundary_fiber << <grid_fiber, threads >> > (d->dev_vel, d->dev_mask, d->dev_rho, d->dev_rho_c, d->dev_f, d->dev_f_c, d->dev_w, d->dev_e);*/
		boundary_block << <grid_fiber, threads >> > (d->dev_vel, d->dev_mask, d->dev_rho, d->dev_rho_c, d->dev_f, d->dev_f_c, d->dev_w, d->dev_e);
		cudaDeviceSynchronize();
	}
	vel_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_vel);
	rho_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_rho_c);
	//mask_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_mask);
	mask_img_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_mask);
	vel_ave_to_bitmap << <m / 16, 16 >> > (d->output_bitmap, d->dev_vel);
	rho_ave_to_bitmap << <m / 16, 16 >> > (d->output_bitmap, d->dev_rho_c);
	rho_y_to_bitmap << <n / 16, 16 >> > (d->output_bitmap, d->dev_rho_c);
	//vel_arrow_to_bitmap << <grid2d, threads >> > (d->output_bitmap, d->dev_mask, d->dev_vel);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	++d->frames;
}

//void solve(DataBlock* d) {
//	dim3 grid2d(m / 16, n / 16);
//	dim3 grid3d(m / 16, n / 16, 9);
//	dim3 grid_fiber(m / 16, n / 16, fn);
//	dim3 threads(16, 16);
//	init << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_f_c_old, d->dev_f_c, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho, d->dev_rho_c);
//	for (int k = 0; k < nstep; k++) {
//		/*if (k % 10 == 0) {
//			std::cout << k << std::endl;
//		}*/
//		collision << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_f_c_old, d->dev_f_c, d->dev_w, d->dev_rho, d->dev_rho_c, d->dev_e, d->dev_vel);
//		cudaDeviceSynchronize();
//		stream << <grid3d, threads >> > (d->dev_f_old, d->dev_f, d->dev_f_c_old, d->dev_f_c, d->dev_e);
//		cudaDeviceSynchronize();
//		update << <grid2d, threads >> > (d->dev_rho, d->dev_rho_c, d->dev_f, d->dev_f_c, d->dev_vel, d->dev_e);
//		cudaDeviceSynchronize();
//		boundary_lr << <n / 16, 16 >> > (d->dev_f, d->dev_f_c, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho, d->dev_rho_c);
//		cudaDeviceSynchronize();
//		boundary_tb << <m / 16, 16 >> > (d->dev_f, d->dev_f_c, d->dev_w, d->dev_e, d->dev_vel, d->dev_rho, d->dev_rho_c);
//		cudaDeviceSynchronize();
//		boundary_fiber << <grid_fiber, threads >> > (d->dev_vel, d->dev_mask, d->dev_rho, d->dev_rho_c, d->dev_f, d->dev_f_c, d->dev_w, d->dev_e);
//		cudaDeviceSynchronize();
//	}
//}
void print(float* f) {
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < n ; y++) {
			/*std::cout << vel[y * m*2 + x * 2+0] << "," << vel[y * m * 2 + x * 2 + 1] << std::endl;*/
			for (int k = 0; k < 9; k++) {
				std::cout << f[y * m * 9 + x * 9 + k] << ",";
			}
			std::cout << std::endl;
		}
	}
}
void print_rho(float* rho) {
	for (int x = 0; x < m; x++) {
		//std::cout <<  rho[(n - 1) / 2 * m + x] << std::endl;
		std::cout << rho[x] << std::endl;
	}
}
void anim_exit(DataBlock* d) {
	cudaFree(d->dev_rho);
	cudaFree(d->dev_f);
	cudaFree(d->dev_f_old);
	cudaFree(d->dev_rho_c);
	cudaFree(d->dev_f_c);
	cudaFree(d->dev_f_c_old);
	cudaFree(d->dev_e);
	cudaFree(d->dev_w);
}
void imread(std::string filename,int* mask) {
	std::ifstream imgfile(filename);
	if (!imgfile.is_open())
	{
		std::cout << "can not open this file" << std::endl;
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
	srand((unsigned)time(NULL));
	float w[9] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9.
			, 1. / 36., 1. / 36., 1. / 36., 1. / 36. };
	float e[9 * 2] = { 0, 0, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1,
		-1, 1, -1, -1, 1, -1 };
	float* rho, * f, * f_old, * vel;
	int* mask,*fibers;
	float* rho_c,* f_c, * f_c_old;
	float fluxq[m], flux[m],Tm[m];
	for (int x = 0; x < m; x++) {
		fluxq[x] = 0;
		flux[x] = 0;
		Tm[x] = 0;
	}
	DataBlock data;
	CPUAnimBitmap bitmap((int)m*3, (int)n*2, (void*)&data);
	data.bitmap = &bitmap;
	data.frames = 0;
	rho = new float[m * n];
	f = new float[m * n * 9];
	f_old = new float[m * n * 9];
	mask = new int[m * n];
	fibers = new int[fn * 3];
	vel = new float[m * n*2];

	rho_c = new float[m * n];
	f_c = new float[m * n*9];
	f_c_old = new float[m * n * 9];
	
	for (int i = 0; i < fn;i++) {
		fibers[i * 3 + 0] = rand() % m;
		fibers[i * 3 + 1] = rand() % n;
		fibers[i * 3 + 2] = rand() % 180;
		//std::cout << fibers[i * 3 + 0] <<", "<< fibers[i * 3 + 1] << ", " << fibers[i * 3 + 2] << std::endl;
	}
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < m; x++) {
			mask[y * m + x] = 0;
			vel[y * m * 2 + x * 2 + 0] = 0;
			vel[y * m * 2 + x * 2 + 1] = 0;
			rho[y * m + x] = 1;
			rho_c[y * m + x] = 0;
			for (int k = 0; k < 9; k++) {
				f[y * m * 9 + x * 9 + k] = 0;
				f_old[y * m * 9 + x * 9 + k] = 0;
				f_c[y * m * 9 + x * 9 + k] = 0;
				f_c_old[y * m * 9 + x * 9 + k] = 0;
			}
			if (x > 0 && x < m - 1 && y>0 && y < n - 1) {
				for (int i = 0; i < fn; i++) {
					int fx = fibers[i * 3 + 0];
					int fy = fibers[i * 3 + 1];
					float fd = fibers[i * 3 + 2] / 180.0 * PI;
					if ((pow((float)(x - fx),2.0) + pow((float)(y - fy), 2.0)) * pow(cos(atan((float)(y - fy) / (x - fx)) - fd), 2.0) <= fl * fl / 4.0 && (pow((float)(x - fx), 2.0) + pow((float)(y - fy), 2.0)) * pow(sin(atan((float)(y - fy) / (x - fx)) - fd), 2.0) <= fw * fw / 4.0) {
					//if ((pow((float)(x - fx), 2.0) + pow((float)(y - fy), 2.0))  <= fw * fw) {
						if (x >= fx && y > fy) {
							mask[y * m + x] = 1;
						}
						else if (x < fx && y >= fy) {
							mask[y * m + x] = 2;
						}
						else if (x <= fx && y < fy) {
							mask[y * m + x] = 3;
						}
						else if (x > fx && y <= fy) {
							mask[y * m + x] = 4;
						}
					}
				}
			}
		}
	}
	imread("ketton.txt", mask);
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());
	cudaMalloc((void**)&data.dev_rho, sizeof(float) * m * n);
	cudaMalloc((void**)&data.dev_vel, sizeof(float) * m * n*2);
	cudaMalloc((void**)&data.dev_f, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_f_old, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_rho_c, sizeof(float) * m * n);
	cudaMalloc((void**)&data.dev_f_c, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_f_c_old, sizeof(float) * m * n * 9);
	cudaMalloc((void**)&data.dev_w, sizeof(float) * 9);
	cudaMalloc((void**)&data.dev_e, sizeof(float) * 9 * 2);
	cudaMalloc((void**)&data.dev_mask, sizeof(int) * m * n);

	cudaMemcpy(data.dev_rho, rho, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_vel, vel, sizeof(float) * m * n*2, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f, f, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f_old, f_old, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_rho_c, rho_c, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f_c, f_c, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_f_c_old, f_c_old, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_w, w, sizeof(float) * 9, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_e, e, sizeof(float) * 9 * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(data.dev_mask, mask, sizeof(int) * m * n, cudaMemcpyHostToDevice);
	
	data.bitmap->anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);

	//std::string filename = "./lbm_output"+ std::to_string((unsigned)time(NULL)) +".gif";
	//GifWriter writer = {};
	//GifBegin(&writer, filename.data(), 3 * m, 2 * n, 20, 8, true);
	//unsigned char* image;
	//image = new unsigned char[3 * m * 2 * n * 4];
	//unsigned char* image_rev;
	//image_rev = new unsigned char[3 * m * 2 * n * 4];
	//for (int frame = 0; frame < 100; ++frame)
	//{
	//	printf("Writing frame %d...\n", frame);
	//	anim_gpu(&data, frame);
	//	cudaMemcpy(image, data.output_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	//	for (int y = 0; y < 2*n; y++) {
	//		for (int x = 0; x < 3 * m; x++) {
	//			image_rev[(y * 3 * m + x)*4+0] = image[((2*n-y-1) * 3 * m + x)*4+0];
	//			image_rev[(y * 3 * m + x) * 4 + 1] = image[((2 * n - y - 1) * 3 * m + x) * 4 + 1];
	//			image_rev[(y * 3 * m + x) * 4 + 2] = image[((2 * n - y - 1) * 3 * m + x) * 4 + 2];
	//			image_rev[(y * 3 * m + x) * 4 + 3] = image[((2 * n - y - 1) * 3 * m + x) * 4 + 3];
	//		}
	//	}
	//		
	//	GifWriteFrame(&writer, image_rev, 3*m, 2*n, 2, 8, true);
	//}
	//GifEnd(&writer);

	//solve(&data);
	/*cudaMemcpy(f_c, data.dev_f_c, sizeof(float) * m * n*9, cudaMemcpyDeviceToHost);
	print(f_c);*/
	//cudaMemcpy(rho_c, data.dev_rho_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
	//print_rho(rho_c);
	
	delete[] f;
	delete[] f_old;
	delete[] rho;
	delete[] vel;
	delete[] f_c;
	delete[] f_c_old;
	delete[] rho_c;
}