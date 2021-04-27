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

void collision(float* f_old, float* f, float* rho) {
	for (int x = 0; x < m; x++) {
		for (int k = 0; k < 2; k++) {
			f_old[x * 2 + k] = (1 - omega) * f[x * 2 + k] + omega * 0.5 * rho[x];
		}
	}
}
void stream(float* f_old, float* f) {
	for (int x = 0; x < m-1; x++) {
		f[(m - x - 1) * 2 + 0] = f_old[(m - x - 2) * 2 + 0];
		f[x * 2 + 1] = f_old[(x + 1) * 2 + 1];
	}	
}
void boundary(float* f) {
	f[0 * 2 + 0] = twall - f[0 * 2 + 1];
	f[(m-1) * 2 + 0] = f[(m - 2) * 2 + 0];
	f[(m - 1) * 2 + 1] = f[(m - 2) * 2 + 1];
}

void update(float* rho,float* f) {
	for (int x = 0; x < m; x++) {
		rho[x] = f[x * 2 + 0] + f[x * 2 + 1];
	}
}
void rho_to_bitmap(unsigned char* bitmap, float* rho) {
	for (int x = 0; x < m; x++) {
		for (int y = 0; y < m; y++) {
			int offset = x + y * m;
			if (y <= rho[x] * (m - 50)) {
				bitmap[offset * 4 + 0] = 255;
				bitmap[offset * 4 + 1] = 0;
				bitmap[offset * 4 + 2] = 0;
				bitmap[offset * 4 + 3] = 255;
			}
		}
	}
	
}
void anim_gpu(DataBlock* d, int ticks) {
	CPUAnimBitmap* bitmap = d->bitmap;
	for (int i = 0; i < nstep; i++) {
		collision(d->dev_f_old, d->dev_f, d->dev_rho);
		stream(d->dev_f_old, d->dev_f);
		boundary(d->dev_f);
		update(d->dev_rho, d->dev_f);
	}
	rho_to_bitmap(d->output_bitmap, d->dev_rho);
	memcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size());
	++d->frames;
}
void solve(DataBlock* d) {
	for (int i = 0; i < nstep; i++) {
		collision(d->dev_f_old, d->dev_f, d->dev_rho);
		stream(d->dev_f_old, d->dev_f);
		boundary(d->dev_f);
		update(d->dev_rho, d->dev_f);
	}
}
void anim_exit(DataBlock* d) {
	delete[] d->dev_f;
	delete[] d->dev_f_old;
	delete[] d->dev_rho;
	delete[] d->bitmap;
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
	data.output_bitmap = (unsigned char*)malloc(bitmap.image_size());
	data.dev_rho = rho;
	data.dev_f = f;
	data.dev_f_old = f_old;

	/*data.bitmap->anim_and_exit((void(*)(void*, int))anim_gpu, (void(*)(void*))anim_exit);*/
	solve(&data);

	for (int x = 0; x < m; x++) {
		std::cout << rho[x] << ",";
	}
	std::cout << std::endl;

	
}