#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#define m 201
#define n 201
class lbm_solver {
	float xl = 1.0;
	float yl = 1.0;
	float dx = xl / (m - 1);
	float dy = yl / (n - 1);
	float w[9] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9.
			, 1. / 36., 1. / 36., 1. / 36., 1. / 36. };
	float e[9][2]= {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1},
		{-1, 1}, {-1, -1}, {1, -1}};
	float c2 = 1.0 / 3.0;
	float *rho,*f,*f_old,*Z;
	float flux[m];
	float fluxq[m];
	float Tm[m];
	float x[m];
	float y[n];
	float alpha = 0.25;
	float omega = 1.0 / (3.0 * alpha + 0.5);
	float twall = 1.0;
	int nstep = 2000;
public:
	lbm_solver() {
		rho = new float[m * n];
		Z = new float[m * n];
		f = new float[m * n * 9];
		f_old = new float[m * n * 9];
		for (int i = 1; i < m; i++) {
			x[i] = x[i - 1] + dx;
			y[i] = y[i - 1] + dy;
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				rho[i*n+j] = 0;
				for (int k = 0; k < 9; k++) {
					f[i*n*9+j*9+k] = 0;
					f_old[i * n * 9 + j * 9 + k] = 0;
				}
			}
		}
	}
	float feq(int i, int j, int k) {
		return w[k] * rho[i*n + j];
	}
	void collision() {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				for (int k = 0; k < 9; k++) {
					f_old[i * n * 9 + j * 9 + k] = (1 - omega) * f[i * n * 9 + j * 9 + k] + omega * feq(i, j, k);
				}
			}
		}
	}
	void stream() {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				for (int k = 0; k < 9; k++) {
					int ip = i - e[k][0];
					int jp = j - e[k][1];
					if (ip>=0 && ip < m && jp>=0 && jp < n) {
						f[i * n * 9 + j * 9 + k] = f_old[ip * n * 9 + jp * 9 + k];
					}
				}
			}
		}
	}
	void boundary() {
		for (int j = 0; j < n; j++) {
			f[0 * n * 9 + j * 9 + 1] = w[1] * twall + w[3] * twall - f[0 * n * 9 + j * 9 + 3];
			f[0 * n * 9 + j * 9 + 5] = w[5] * twall + w[7] * twall - f[0 * n * 9 + j * 9 + 7];
			f[0 * n * 9 + j * 9 + 8] = w[8] * twall + w[6] * twall - f[0 * n * 9 + j * 9 + 6];
		}
		for (int i = 0; i < m; i++) {
			f[i * n * 9 + 0 * 9 + 2] = f[i * n * 9 + 1 * 9 + 2];
			f[i * n * 9 + 0 * 9 + 5] = f[i * n * 9 + 1 * 9 + 5];
			f[i * n * 9 + 0 * 9 + 6] = f[i * n * 9 + 1 * 9 + 6];
		}
		for (int i = 0; i < m; i++) {
			f[i * n * 9 + (n-1) * 9 + 7] = -f[i * n * 9 + (n - 1) * 9 + 5];
			f[i * n * 9 + (n - 1) * 9 + 4] = -f[i * n * 9 + (n - 1) * 9 + 2];
			f[i * n * 9 + (n - 1) * 9 + 8] = -f[i * n * 9 + (n - 1) * 9 + 6];
		}
		for (int j = 0; j < n; j++) {
			f[(m - 1) * n * 9 + j * 9 + 3] = -f[(m-1) * n * 9 + j * 9 + 1];
			f[(m - 1) * n * 9 + j * 9 + 7] = -f[(m - 1) * n * 9 + j * 9 + 5];
			f[(m - 1) * n * 9 + j * 9 + 6] = -f[(m - 1) * n * 9 + j * 9 + 8];
		}
	}
	void update() {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				rho[i * n + j] = f[i * n * 9 + j * 9 + 0];
				for (int k = 1; k < 9; k++) {
					rho[i * n + j] += f[i * n * 9 + j * 9 + k];
				}
			}
		}
		for (int i = 0; i < m; i++) {
			Tm[i] = rho[i * n + (n - 1) / 2];
		}
	}
	void post() {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				Z[j * n + i] = rho[i * n + j];
			}
		}
	}
	void solve() {
		for (int k = 0; k < nstep; k++) {
			collision();
			stream();
			boundary();
			update();
			post();
		}
	}
	void plot() {
		for (int i = 0; i < m; i++) {
			std::cout << Tm[i] << std::endl;
		}
	}
};

int main() {
	lbm_solver lbm;
	lbm.solve();
	lbm.plot();
}