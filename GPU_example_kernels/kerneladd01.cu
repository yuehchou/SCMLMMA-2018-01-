#include <stdlib.h>
#include <stdio.h>

// Kernel adding entries of the adjacent array entries (radius of 3) of a 1D array
//
// initial approach
// * 7 kernels, each adding one element to the sum
// * data always read from main memory

__global__ void kernel_add(int n, int offset, int *a, int *b)
{
   int i = blockDim.x*blockIdx.x+threadIdx.x;
   int j = i + offset;
   if( j>-1 && j<n ){
        b[i]+=a[j];
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

  cudaEventRecord(start);
  
  dim3 blocksize(256); 
  dim3 gridsize((n+blocksize.x-1)/(blocksize.x));
  
  kernel_add<<<gridsize, blocksize>>>(n, -3, d_a, d_b);
  kernel_add<<<gridsize, blocksize>>>(n, -2, d_a, d_b);
  kernel_add<<<gridsize, blocksize>>>(n, -1, d_a, d_b);
  kernel_add<<<gridsize, blocksize>>>(n, 0, d_a, d_b);
  kernel_add<<<gridsize, blocksize>>>(n, 1, d_a, d_b);
  kernel_add<<<gridsize, blocksize>>>(n, 2, d_a, d_b);
  kernel_add<<<gridsize, blocksize>>>(n, 3, d_a, d_b);
  
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