#include"stdio.h"
#include"stdlib.h"
#include<math.h>
#include <sys/time.h>    
#define USECUDA 1
#if USECUDA==1
#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#endif
#define Dev_Loop 1024
#define BlockN 1024
#define addNum 16
//这个程序由CUDA并行构架编写而成
long getCurrentTime()  
{  
	struct timeval tv;  
	gettimeofday(&tv,NULL);  
	return tv.tv_sec * 1000000 + tv.tv_usec;  
}
double randomf()//c 的随机数产生器改为产生小数
{
	return ((double)rand())/RAND_MAX;
}
#if USECUDA==1
//以下代码全部在CUDA构架下运行的
__global__ void inte_cell(double* l,double* r,double *res,long *time,curandState *state)
{
	//不要传函数指针
	int i = blockIdx.x*BlockN+threadIdx.x;
	long seed=(*time)+(i);//因为所有给定时间一定，所以我们只能通过对时间进行简单处理
	int offset=0;//完全独立的序列，所以offset全部为零来节约时间
	curand_init (seed,i,offset,&state[i]);//设置第i个随机序列
	double x=1,sum=0;
	double k=8;
	for(int j=0;j<k*Dev_Loop;j++)
	{
		x=(r[i]-l[i])*curand_uniform_double(&state[i]);
		sum+=sqrt(x+sqrt(x));//func(x);
	}
	res[i]=sum/(Dev_Loop*k);
	__syncthreads();

}
__global__ void big_plus(double*a,double *res,int *threadNum)
{
	//为了尽可能的利用并行效率，加法采用两次树形相加的形式，每次加addNum个
	//如此可以对付2^30次的快速相加
	//虽然嘛。。。。这是毫无意义的啦！因为本程序只有2^20次的相加
	//不过留个接口以后用总是好事情

	int i=blockIdx.x*(*threadNum)+threadIdx.x;
	double sum=0;
	int k=i*addNum;
	for(int j=0;j<addNum;j++)
	{
		sum+=a[k];
		k++;
	}
	res[i]=sum/addNum;
	__syncthreads();
	
}
__device__ double func0(double x)
{
	//被积分函数0,实际应用仅需修改此函数即可
	//同时考虑了代码复用性
	return sqrt(x+sqrt(x));
}
long*getCurrentTimeForDev()
{	
	long *time;
	cudaMalloc(&time,sizeof(long));
	long *timenow=new long;
	*timenow=getCurrentTime();
	cudaMemcpy(time,timenow,sizeof(long),cudaMemcpyHostToDevice);
	return time;
}
double *DevValueD(double v,int len)//把host值转化为dev指针值
{
	double*res;
	cudaMalloc(&res,sizeof(double)*len);
	double *val=new double[len];
	for(int i=0;i<len;i++)
		val[i]=v;
	cudaMemcpy(res,val,sizeof(double)*len,cudaMemcpyHostToDevice);
	return res;
}
int *DevValueI(int v,int len)
{
	int*res;
	cudaMalloc(&res,sizeof(int)*len);
	int *val=new int[len];
	for(int i=0;i<len;i++)
		val[i]=v;
	cudaMemcpy(res,val,sizeof(int)*len,cudaMemcpyHostToDevice);
	return res;
}
#endif

double inte_cell_cpu(double l,double r)
{
	double x;
	double res=0;
	for(int i=0;i<Dev_Loop;i++)
	{
		x=(r-l)*randomf();
		res+=sqrt(x+sqrt(x*x));
	}
	res/=Dev_Loop;
	return res;
}


int work()
{
	int threadPerBlock=BlockN;
	int numBlocks= 256;
	size_t size = BlockN *numBlocks*sizeof(double);

	long st=getCurrentTime();

	curandState *state;
	cudaMalloc(&state,sizeof(curandState)*1024*1024);//设立随机状态列

	double* d_A,*add_tem0,*add_tem1,*res;
	cudaMalloc(&d_A, size);
	cudaMalloc(&add_tem0, size/16);
	cudaMalloc(&add_tem1, size/256);
	cudaMalloc(&res,sizeof(double));

	inte_cell<<<numBlocks,threadPerBlock>>>(DevValueD(0.0,numBlocks*threadPerBlock),DevValueD(1.0,numBlocks*threadPerBlock),d_A,getCurrentTimeForDev(),state);
	/*
	big_plus<<<numBlocks/addNum,threadPerBlock>>>(d_A,add_tem0,DevValueI(1024,numBlocks*threadPerBlock));
	big_plus<<<numBlocks/addNum/addNum,threadPerBlock>>>(add_tem0,add_tem1,DevValueI(1024,numBlocks*threadPerBlock/addNum));
*/
	double *result=new double[numBlocks*threadPerBlock];
	/*
	FILE * out0,*out1,*out2;
	out0=fopen("data0.txt","w");
	out1=fopen("data1.txt","w");
	out2=fopen("data2.txt","w");


	cudaMemcpy(result, d_A, size, cudaMemcpyDeviceToHost);
	for(int i=0;i<1024;i++)
		fprintf(out0,"%f\n",result[i]);

	cudaMemcpy(result, add_tem0, size/addNum, cudaMemcpyDeviceToHost);
	for(int i=0;i<1024;i++)
		fprintf(out1,"%f\n",result[i]);
	*/
	/*
	for(int i=0;i<numBlocks*threadPerBlock;i++)
		fprintf(out0,"%f\n",result[i]);
		*/
	double fin_res=0;
	cudaMemcpy(result,d_A, size, cudaMemcpyDeviceToHost);
	for(int i=0;i<numBlocks*threadPerBlock;i++)
	{
		fin_res+=result[i];
	}
	fin_res/=(numBlocks*threadPerBlock);
	long ed=getCurrentTime();

	printf("GPU running Time:%ld\n",ed-st);
	printf("final:%16.14f\n",fin_res);

	/*
	st=getCurrentTime();
	double sum=0;
	for(int i=0;i<256;i++)
		for(int j=0;j<1024;j++)
		{
			sum+=inte_cell_cpu(0,1);
		}
	sum/=(1024*256);
	ed=getCurrentTime();
	printf("cpu:time:%ld,res:%15f\n",ed-st,sum);
*/
	/*
	fclose(out0);
	fclose(out1);
	fclose(out2);
	*/
	cudaFree(d_A);
	cudaFree(d_A);
	cudaFree(add_tem0);
	cudaFree(add_tem1);
	cudaFree(state);
	cudaFree(res);
}
int main()
{
	work();
}
