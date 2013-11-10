#include"stdio.h"
#include<cuda_runtime.h>
#include <sys/time.h>    

#define N 1024
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	for(int j=0;j<1000;j++)
		C[i] = (A[i] * B[i]);
}

long getCurrentTime()  
{  
	struct timeval tv;  
	gettimeofday(&tv,NULL);  
	return tv.tv_sec * 1000000 + tv.tv_usec;  
}
void cpu_VecAdd(int i,float* A, float* B, float* C)
{
	for(int j=0;j<1000;j++)
		C[i] = (A[i] * B[i]);
}
int main()
{
	// Kernel invocation with N threads
	printf("Hello,World\n");
	float *A=new float[N],*B=new float[N],*C=new float[N];
	for(int i=0;i<N;i++)
	{
		A[i]=i;
		B[i]=2*i;
	}

	size_t size = N * sizeof(float);

	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	float *e=new float[N];
	long st=getCurrentTime();
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	VecAdd<<<1, N>>>(d_A, d_B, d_C);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	long ed=getCurrentTime();
	printf("gpu running time:%ld\n",ed-st);
	st=getCurrentTime();
	for(int i=0;i<N;i++)
		cpu_VecAdd(i,A,B,e);
	ed=getCurrentTime();
	printf("cpu running time:%ld\n",ed-st);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	for(int i=0;i<N;i++)
	{
		//printf("%f ",C[i]);
	}
	printf("\n");
}
