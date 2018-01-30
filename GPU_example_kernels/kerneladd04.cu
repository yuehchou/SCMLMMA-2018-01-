#include <stdlib.h>
#include <stdio.h>

// Kernel adding entries of the adjacent array entries (radius of 3) of a 1D array
//
// even better
// * one thread reads needed data into shared memory
// * every thread-block computes blockDim.x partial sums
// * data read from shared memory
// even better
// * every thread reads one entry into shared memory
// * every thread-block computes blockDim.x-6 partial sums
// * data read from shared memory

__global__ void kernel4(int n, int *a, int *b)
{
   int i = (blockDim.x-6)*blockIdx.x+threadIdx.x-3;
   int idx = threadIdx.x;
   __shared__ int values[256];
   int sum = 0;
     
   values[idx] = (i>-1 && i<n ) ? a[i] : 0;
   
   if( idx>2 && idx<256-3 ){
     for( int j=-3; j<4; j++)
       sum += values[ idx+j ];       
   
       
     b[i]=sum;
   }
}

int main() {


  int n=2000000;
  int memSize = n*sizeof(int);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int *a, *d_a;
  a = (int*) malloc (n*sizeof(*a));
  cudaMalloc( (void**) &d_a, memSize);
  int *b, *d_b;
  b = (int*) malloc (n*sizeof(*b));
  cudaMalloc( (void**) &d_b, memSize);
  
  for(int j=0; j<n; j++){
  	a[j] = j;
  	b[j] = 0;
  }

  cudaMemcpy( d_a, a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, b, memSize, cudaMemcpyHostToDevice);

  dim3 block(256); 
  dim3 grid((n+block.x-7)/(block.x-6));
  
  cudaEventRecord(start);
  kernel4<<<grid,block>>>(n,d_a,d_b);
  cudaEventRecord(stop);

  cudaMemcpy( b, d_b, memSize, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("runtime [s]: %f\n", milliseconds/1000.0);
  
  for(int j=0; j<10; j++)
  	printf("%d\n",b[j]);
  	
  cudaFree(d_a);
  free(a);
  cudaFree(d_b);
  free(b);
  
  return 0;

}