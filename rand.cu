#include"stdio.h"
#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include <sys/time.h>    

#define N 1024
// Kernel definition
__global__ void random_gpu(long* C,long* time,curandState*state)
{
	long i = threadIdx.x;
	long seed=(*time)*(i+1);//因为所有给定时间一定，所以我们只能通过对时间进行简单处理
	int offset=0;//完全独立的序列，所以offset全部为零来节约时间
	curand_init (seed,i,offset,&state[i]);//设置第i个随机序列
	C[i]=curand(&state[i]);//获得第i个随机序列的随机值
}

long getCurrentTime()  
{  
	struct timeval tv;  
	gettimeofday(&tv,NULL);  
	return tv.tv_sec * 1000000 + tv.tv_usec;  
}
long*getCurrentTimeForDev()
{	long *time;
	cudaMalloc(&time,sizeof(long));
	long *timenow=new long;
	*timenow=getCurrentTime();
	cudaMemcpy(time,timenow,sizeof(long),cudaMemcpyHostToDevice);
	return time;
}
int main()
{
	size_t size = N * sizeof(float);

	long* C=new long[N];
	long st=getCurrentTime();
	curandState *state;
	long *d_C;
	cudaMalloc(&state,sizeof(curandState)*N);//设立随机状态列
	cudaMalloc(&d_C, size);
	random_gpu<<<1,N>>>(d_C,getCurrentTimeForDev(),state);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	long ed=getCurrentTime();
	printf("gpu running time:%ld\n",ed-st);
	cudaFree(d_C);
	for(int i=0;i<10;i++)
	{
		printf("%ld ",C[i]);
	}
	delete[] C;
	printf("\n");
}
