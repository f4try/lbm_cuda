#include <iostream>
#define m 256
#define n 256
#define alpha 0.25f
#define omega (1.0 / (3.0 * alpha + 0.5))
#define twall 1.0f
#define nstep 200


void collision(float* f_old, float* f, float* w, float* rho) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < m; x++) {
			for (int k = 0; k < 9; k++) {
				f_old[y * m * 9 + x * 9 + k] = (1 - omega) * f[y * m * 9 + x * 9 + k] + omega * w[k] * rho[y * m + x];
			}
		}
	}
}
void stream(float* f_old, float* f, float* e) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < m; x++) {
			for (int k = 0; k < 9; k++) {
				int xp = x - e[k * 2 + 0];
				int yp = y - e[k * 2 + 1];
				if (xp >= 0 && xp < m && yp >= 0 && yp < n) {
					f[y * m * 9 + x * 9 + k] = f_old[yp * m * 9 + xp * 9 + k];
				}
			}
		}
	}
}
void boundary_tb(float* f) {
	for (int x = 0; x < m; x++) {
		f[0 * m * 9 + x * 9 + 2] = f[1 * m * 9 + x * 9 + 2];
		f[0 * m * 9 + x * 9 + 5] = f[1 * m * 9 + x * 9 + 5];
		f[0 * m * 9 + x * 9 + 6] = f[1 * m * 9 + x * 9 + 6];

		f[(n - 1) * m * 9 + x * 9 + 7] = -f[(n - 1) * m * 9 + x * 9 + 5];
		f[(n - 1) * m * 9 + x * 9 + 4] = -f[(n - 1) * m * 9 + x * 9 + 2];
		f[(n - 1) * m * 9 + x * 9 + 8] = -f[(n - 1) * m * 9 + x * 9 + 6];
	}
}
void boundary_lr(float* f, float* w) {
	for (int y = 0; y < n; y++) {
		f[y * m * 9 + 0 * 9 + 1] = w[1] * twall + w[3] * twall - f[y * m * 9 + 0 * 9 + 3];
		f[y * m * 9 + 0 * 9 + 5] = w[5] * twall + w[7] * twall - f[y * m * 9 + 0 * 9 + 7];
		f[y * m * 9 + 0 * 9 + 8] = w[8] * twall + w[6] * twall - f[y * m * 9 + 0 * 9 + 6];

		f[y * m * 9 + (m - 1) * 9 + 3] = -f[y * m * 9 + (m - 1) * 9 + 1];
		f[y * m * 9 + (m - 1) * 9 + 7] = -f[y * m * 9 + (m - 1) * 9 + 5];
		f[y * m * 9 + (m - 1) * 9 + 6] = -f[y * m * 9 + (m - 1) * 9 + 8];
	}
}
void update(float* rho, float* f) {
	for (int y = 0; y < n; y++) {
		for (int x = 0; x < m; x++) {
			rho[y * m + x] = f[y * m * 9 + x * 9 + 0] + f[y * m * 9 + x * 9 + 1] + f[y * m * 9 + x * 9 + 2] + f[y * m * 9 + x * 9 + 3] + f[y * m * 9 + x * 9 + 4] + f[y * m * 9 + x * 9 + 5] + f[y * m * 9 + x * 9 + 6] + f[y * m * 9 + x * 9 + 7] + f[y * m * 9 + x * 9 + 8];
		}
	}
}

class lbm_solver {
	float w[9] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9.
			, 1. / 36., 1. / 36., 1. / 36., 1. / 36. };
	float e[9 * 2] = { 0, 0, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1,
		-1, 1, -1, -1, 1, -1 };
	float* rho, * f, * f_old;
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

	}
	~lbm_solver() {
		delete[] f;
		delete[] f_old;
		delete[] rho;
	}


	void post() {
		for (int x = 0; x < m; x++) {
			Tm[x] = rho[(n - 1) / 2 * m + x];
		}
	}
	void solve() {
		for (int k = 0; k < nstep; k++) {
			collision  (f_old, f, w, rho);
			stream (f_old, f, e);
			boundary_tb  (f);
			boundary_lr (f, w);
			update  (rho, f);
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