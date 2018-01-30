#include <stdlib.h>
#include <stdio.h>


__global__ void kernel11(int *a, int *b, int *c)
{
   a[blockIdx.y*blockDim.x*gridDim.x+blockIdx.x*blockDim.x + threadIdx.x]=blockIdx.x;
   b[blockIdx.y*blockDim.x*gridDim.x+blockIdx.x*blockDim.x + threadIdx.x]=blockIdx.y;
   c[blockIdx.y*blockDim.x*gridDim.x+blockIdx.x*blockDim.x + threadIdx.x]=threadIdx.x;
}

int main() {


  int n=24;
  int memSize = n*sizeof(int);

  int *a, *b, *c, *d_a, *d_b, *d_c;
  a = (int*) malloc (n*sizeof(*a));
  b = (int*) malloc (n*sizeof(*b));
  c = (int*) malloc (n*sizeof(*c));
  cudaMalloc( (void**) &d_a, memSize);
  cudaMalloc( (void**) &d_b, memSize);
  cudaMalloc( (void**) &d_c, memSize);


  cudaMemcpy( d_a, a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, b, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_c, c, memSize, cudaMemcpyHostToDevice);

  int d1=4;
  int d2=2;
  int d3=1;
  int db=3;
  
  dim3 block(db); 
  dim3 grid(d1, d2, d3);
  kernel11<<<grid,block>>>(d_a, d_b, d_c);

  cudaMemcpy( a, d_a, memSize, cudaMemcpyDeviceToHost);
  cudaMemcpy( b, d_b, memSize, cudaMemcpyDeviceToHost);
  cudaMemcpy( c, d_c, memSize, cudaMemcpyDeviceToHost);
  
  for(int l=0; l<n; l++)
          printf("(%d, %d) -> %d\n",a[l], b[l], c[l]);
       
          printf("\n\n\n\n 2D output:\n\n");
          
 for(int k=0;k<d2; k++){
    for(int i=0; i<db; i++){
      for(int j=0; j<d1; j++){
        int l = j*db+i+k*db*d1;
        printf("(%d, %d) -> %d    ",a[l], b[l], c[l]);
      }
      printf("\n");
    }
    printf("\n");
  }
  
  cudaFree(d_a);
  free(a);
  cudaFree(d_b);
  free(b);
  cudaFree(d_c);
  free(c);
  
  return 0;

}