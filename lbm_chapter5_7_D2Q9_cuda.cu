#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#define m 256
#define n 256
#define alpha 0.25f
#define omega (1.0 / (3.0 * alpha + 0.5))
#define twall 1.0f
#define nstep 2000


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

class lbm_solver {
	float w[9] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9.
			, 1. / 36., 1. / 36., 1. / 36., 1. / 36. };
	float e[9 * 2] = { 0, 0, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1,
		-1, 1, -1, -1, 1, -1 };
	float* rho, * f, * f_old;
	float* dev_rho, * dev_f, * dev_f_old, * dev_w, * dev_e;
	float Tm[m];
public:
	lbm_solver() {
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

		cudaMalloc((void**)&dev_rho, sizeof(float) * m * n);
		cudaMalloc((void**)&dev_f, sizeof(float) * m * n * 9);
		cudaMalloc((void**)&dev_f_old, sizeof(float) * m * n * 9);
		cudaMalloc((void**)&dev_w, sizeof(float) * 9);
		cudaMalloc((void**)&dev_e, sizeof(float) * 9 * 2);

		cudaMemcpy(dev_rho, rho, sizeof(float) * m * n, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_f, f, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_f_old, f_old, sizeof(float) * m * n * 9, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_w, w, sizeof(float) * 9, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_e, e, sizeof(float) * 9 * 2, cudaMemcpyHostToDevice);
	}
	~lbm_solver() {
		cudaFree(dev_rho);
		cudaFree(dev_f);
		cudaFree(dev_f_old);
		cudaFree(dev_e);
		cudaFree(dev_w);
		delete[] f;
		delete[] f_old;
		delete[] rho;
	}


	void post() {
		cudaMemcpy(rho, dev_rho, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
		for (int x = 0; x < m; x++) {
			Tm[x] = rho[(n - 1) / 2 * m + x];
		}
	}
	void solve() {
		dim3 grid2d(m / 16, n / 16);
		dim3 grid3d(m / 16, n / 16, 9);
		dim3 threads(16, 16);
		for (int k = 0; k < nstep; k++) {
			collision << <grid3d, threads >> > (dev_f_old, dev_f, dev_w, dev_rho);
			cudaDeviceSynchronize();
			stream << <grid3d, threads >> > (dev_f_old, dev_f, dev_e);
			cudaDeviceSynchronize();
			boundary_tb << <n / 16, 16 >> > (dev_f);
			cudaDeviceSynchronize();
			boundary_lr << <m / 16, 16 >> > (dev_f, dev_w);
			cudaDeviceSynchronize();
			update << <grid2d, threads >> > (dev_rho, dev_f);
			cudaDeviceSynchronize();
		}
		post();
	}
	void plot() {
		/*for (int y = 0; y < m; y++) {
			for (int x = 0; x < 2; x++) {
				std::cout << rho[y * m + x] << ",";
			}
			std::cout << std::endl;
		}*/
		for (int x = 0; x < m; x++) {
			std::cout <<  Tm[x] << std::endl;
		}
	}
};

int main() {
	lbm_solver lbm;
	lbm.solve();
	lbm.plot();
}