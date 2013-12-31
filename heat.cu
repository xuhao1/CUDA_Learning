#include"stdio.h"
#include<cuda_runtime.h>
#include <sys/time.h>    

#define len 1
#define WIDTH 128
// Kernel definition
__device__ float& getPos(float *T,int x,int y,int w)
{
	return *(T+y*w+x);
}
// 处理：正方形，二维热流场
//          dN
//      dW  dT  dE
//		    dS
__global__ void Calc_Cell(float* T0)
{
	float dW,dE,dN,dS,dT;
	int x=blockIdx.x+1;
	int y=threadIdx.x+1;
	int w=WIDTH;
	for(int i=0;i<100000;i++)
	{
		dN=(getPos(T0,x,y,w)-getPos(T0,x,y-1,w))/len;
		dS=(getPos(T0,x,y+1,w)-getPos(T0,x,y,w))/len;
		dW=(getPos(T0,x,y,w)-getPos(T0,x-1,y,w))/len;
		dE=(getPos(T0,x+1,y,w)-getPos(T0,x,y,w))/len;
		dT=((dS-dN)/len+(dE-dW)/len)*0.1;
		__syncthreads();
		getPos(T0,x,y,w)=getPos(T0,x,y,w)+dT;
	}
}

//储存是行主序，然而坐标是列主序
long getCurrentTime()  
{  
	struct timeval tv;  
	gettimeofday(&tv,NULL);  
	return tv.tv_sec * 1000000 + tv.tv_usec;  
}

int main()
{
	FILE *fp=fopen("a.txt","w");
	size_t size=128*WIDTH*sizeof(float);
	float*d_A;
	cudaMalloc(&d_A, size);
	float A[128*WIDTH]={0};
	for(int i=0;i<128;i++)
		A[i]=100;
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	//在这个计算中我们给定边界条件，仅仅考虑内部的运行状态
	Calc_Cell<<<126,126>>>(d_A);
	cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
	for(int i=0;i<128;i++)
	{
		for(int j=0;j<WIDTH;j++)
		{
			fprintf(fp,"%d %d %f \n",i,j,A[i*WIDTH+j]);
		}
	}
	cudaFree(d_A);
}
