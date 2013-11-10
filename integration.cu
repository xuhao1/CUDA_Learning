#include"stdio.h"
#include"stdlib.h"
#include<math.h>
#include <sys/time.h>    
#define USECUDA 1
#include<cuda_runtime.h>

#define Dev_Loop 1024
#define BlockN 1024
#define addNum 16
//这个程序由CUDA并行构架编写而成
//如果CUDA构架不够了解师兄也可以只看我的host代码部分,只需要把USECUDA 改为 0即可
long getCurrentTime()  
{  
	struct timeval tv;  
	gettimeofday(&tv,NULL);  
	return tv.tv_sec * 1000000 + tv.tv_usec;  
}
__device__ double randomf()//c 的随机数产生器改为产生小数
{
	return (double)(rand()/RAND_MAX);
}
#if USECUDA==1
//以下代码全部在CUDA构架下运行的
__global__ void inte_cell(double l,double r,double *res,double func(double))
{
	int i = blockIdx.x*BlockN+threadIdx.x;
	double x,sum=0;
	res[i]=0;
	for(int i=0;i<Dev_Loop;i++)
	{
		x=(r-l)*randomf();
		sum+=func(x);
	}
	res[i]=sum/Dev_Loop;
	__syncthreads();

}
__global__ void big_plus(double*a,double *res,int threadNum)
{
	//为了尽可能的利用并行效率，加法采用两次树形相加的形式，每次加addNum个
	//如此可以对付2^30次的快速相加
	//虽然嘛。。。。这是毫无意义的啦！因为本程序只有2^20次的相加
	//不过留个接口以后用总是好事情
	int i=blockIdx.x*threadNum+threadIdx.x;
	double sum=0;
	int k=i*addNum;
	for(int j=0;j<addNum;j++)
	{
		sum+=a[k];
		k++;
	}
	res[i]=sum/addNum;
	
}
__global__ void final_plus(double *a,double *res,int r)
{
	double s=0;
	for(int i=0;i<r;i++)
	{
		s+=a[i];
	}
	*res=s/r;
}
__device__ double func0(double x)
{
	//被积分函数0,实际应用仅需修改此函数即可
	//同时考虑了代码复用性
	return sqrt(x+sqrt(x));
}
#endif

double inte_cell_cpu(double l,double r,double func(double))
{
	double x;
	double res=0;
	for(int i=0;i<Dev_Loop;i++)
	{
		x=(r-l)*randomf();
		res+=func(x);
	}
	res/=Dev_Loop;
	return res;
}


int main()
{
	int threadPerBlock=1024;
	int numBlocks= BlockN;
	size_t size = 1024 *1024*1024* sizeof(double);
	long st=getCurrentTime();
	double* d_A,*add_tem0,*add_tem1,*res;
	cudaMalloc(&d_A, size);
	cudaMalloc(&add_tem0, size/16);
	cudaMalloc(&add_tem1, size/256);
	cudaMalloc(&res,sizeof(double));
	inte_cell<<<threadPerBlock,numBlocks>>>(0,1,d_A,func0);
	big_plus<<<1024/addNum,1024>>>(d_A,add_tem0,1024);
	big_plus<<<1024/addNum/addNum,1024>>>(add_tem0,add_tem1,1024);
	final_plus<<<1,1>>>(add_tem1, res ,1024*1024/addNum/addNum);
	double *result=new double;
	cudaMemcpy(res, result, size, cudaMemcpyDeviceToHost);
	long ed=getCurrentTime();
	
}
