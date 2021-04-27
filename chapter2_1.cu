#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void VecAdd(float* A,float* B,float* C){
    int i = threadIdx.x
    C[i] = A[i] + B[i]
}
int main(){
    VecAdd<<<1,N>>>(A,B,C);
}