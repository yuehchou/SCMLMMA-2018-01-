#include <stdlib.h>
#include <stdio.h>

// Kernel adding entries of the adjacent array entries (radius of 3) of a 1D array
//
// better approach
// * merge the 7 kernels into one

__global__ void kernel2(int n, int *a, int *b)
{
   int i = blockDim.x*blockIdx.x+threadIdx.x;
   
   if( i<n ){
     if(i>2)
        b[i]+=a[i-3];
     if(i>1)
        b[i]+=a[i-2];
     if(i>0)
        b[i]+=a[i-1];
     
     b[i]+=a[i]; 
     
     if(i<n-3)
        b[i]+=a[i+3];
     if(i<n-2)
        b[i]+=a[i+2];
     if(i<n-1)
        b[i]+=a[i+1];
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
  dim3 grid((n+block.x-1)/(block.x));
  
  cudaEventRecord(start);
  kernel2<<<grid,block>>>(n,d_a,d_b);
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