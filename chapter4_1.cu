#include <iostream>

#define N 10
void add(int* a, int* b, int* c) {
	int tid = 0;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 1;
	}
}
int main() {
	int a[N], b[N], c[N];
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}
	add(a, b, c);
	for (int i = 0; i < N; i++) {
		printf("%d+%d=%d\n", a[i], b[i], c[i]);
	}
	return 0;
}