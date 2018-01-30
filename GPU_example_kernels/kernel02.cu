#include <stdlib.h>
#include <stdio.h>


__global__ void kernel1(int *a)
{
   if(threadIdx.x > 2 )
        a[blockIdx.x*blockDim.x + threadIdx.x]=100;
    else
      a[blockIdx.x*blockDim.x + threadIdx.x]=blockIdx.x;  
}

int main() {


  int n=20;
  int memSize = n*sizeof(int);

  int *a, *d_a;
  a = (int*) malloc (n*sizeof(*a));
  cudaMalloc( (void**) &d_a, memSize);


  cudaMemcpy( d_a, a, memSize, cudaMemcpyHostToDevice);

  dim3 block(4); 
  dim3 grid(n/block.x);
  kernel1<<<grid,block>>>(d_a);

  cudaMemcpy( a, d_a, memSize, cudaMemcpyDeviceToHost);

  for(int j=0; j<n; j++)
  	printf("%d\n",a[j]);
  	
  cudaFree(d_a);
  free(a);
  
  return 0;

}