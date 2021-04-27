#include <iostream>
#include <math.h>

void add(int n, float* x, float* y) {
	for (int i = 0; i < n; i++) {
		y[i] = x[i] + y[i];
	}
}
int main(void) {
	int N = 1 << 20;
	float *x = new float[N];
	float *y = new float[N];
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	add(N, x, y);
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
	/*	if (fabs(y[i] - 3.0f) > 0.1) {
			std::cout << "yi: " << y[i] << std::endl;
			std::cout << "i: " << i << std::endl;
		}*/
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout <<"y: "<< y[0] << std::endl;
	std::cout <<"maxError: "<< maxError << std::endl;
	delete[] x;
	delete[] y;
	return 0;
}