#include <stdlib.h>
#include <stdio.h>


  __global__ void 
sgemv_rowmajor(int n, float a, float *m, float *x, float *y){

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (row < n){
        for( int col=0; col<n; col++){
            sum+= m[row*n+col] * x[col];
        }
        y[row] = a*sum;
    }  
}

  __global__ void 
sgemv_colmajor(int n, float a, float *m, float *x, float *y){

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (row < n){
        for( int col=0; col<n; col++){
            sum+= m[col*n+row] * x[col];
        }
        y[row] = a*sum;
    }  
}


int main() {


  int n=2000;
  int memSize = n*sizeof(int);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *a, *d_a;
  a = (float*) malloc (n*sizeof(*a));
  cudaMalloc( (void**) &d_a, memSize);
  float *b, *d_b;
  b = (float*) malloc (n*sizeof(*b));
  cudaMalloc( (void**) &d_b, memSize);
  float *m, *d_m;
  m = (float*) malloc (n*n*sizeof(*b));
  cudaMalloc( (void**) &d_m, memSize*n);
  
  for(int j=0; j<n; j++){
  	a[j] = (float) j;
  	b[j] = (float) 0;
  	for(int k=0; k<n; k++)
  	    m[j*n+k] = (float) j+k;
  }
  
  float p = 1.0;

  cudaMemcpy( d_a, a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, b, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_m, m, memSize*n, cudaMemcpyHostToDevice);

  dim3 block(256); 
  dim3 grid((n+block.x-1)/(block.x));
 
  cudaEventRecord(start);
  sgemv_rowmajor<<<grid,block>>>(n, p, d_m, d_a, d_b);
  cudaEventRecord(stop);

  cudaMemcpy( b, d_b, memSize, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("\n\nruntime row-major sgemv [s]: %f\n", milliseconds/1000.0);
  printf("\nresult:\n");
  for(int j=0; j<10; j++)
  	printf("%f\n",b[j]);
  	
  	
  cudaEventRecord(start);
  sgemv_colmajor<<<grid,block>>>(n, p, d_m, d_a, d_b);
  cudaEventRecord(stop);

  cudaMemcpy( b, d_b, memSize, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("\n\nruntime col-major sgemv[s]: %f\n", milliseconds/1000.0);
  printf("\nresult:\n");
  for(int j=0; j<10; j++)
  	printf("%f\n",b[j]);
  	
  cudaFree(d_a);
  free(a);
  cudaFree(d_b);
  free(b);
  cudaFree(d_m);
  free(m);
  
  return 0;

}