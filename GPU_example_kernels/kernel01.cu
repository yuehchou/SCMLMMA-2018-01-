#include <stdlib.h>
#include <stdio.h>

/*
__global__ void kernel10(int *a)
{
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x); 
}

*/


__global__ void kernel10(int *a)
{
   if (threadIdx.x == 1 )
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
   if (threadIdx.x == 0 )
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
   if (threadIdx.x == 3 )
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
   if (threadIdx.x == 2 )
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);

    
    
}



int main() {


  int n=20;
  int memSize = n*sizeof(int);

  int *a, *d_a;
  a = (int*) malloc (n*sizeof(*a));
  cudaMalloc( (void**) &d_a, memSize);


  cudaMemcpy( d_a, a, memSize, cudaMemcpyHostToDevice);

  dim3 block(4); 
  dim3 grid(3);
  kernel10<<<grid,block>>>(d_a);

  cudaMemcpy( a, d_a, memSize, cudaMemcpyDeviceToHost);

  	
  cudaFree(d_a);
  free(a);
  
  return 0;

}