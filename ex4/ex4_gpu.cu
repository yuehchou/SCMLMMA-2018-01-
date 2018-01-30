#include <stdlib.h>
#include <stdio.h>
#define L 100
#define N 1024

__global__ void kernelcount_nonz(int size, double *A, int *nonz, int *rowptr)
{
	int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x, count = 0;
	int num = bid*bdim + tid;

	if(bid*bdim + tid < size)
    {
		for(int i = 0; i < size; i++)
		{
			if(A[num*size + i] != 0.0)
			{
				count ++;
			}
		}
		nonz[num] = count;

		__syncthreads();

		count = 0;
		for(int i = 0; i <= num; i++)
		{
			count += nonz[i];
		}
		rowptr[num + 1] = count;
	}
}

__global__ void kernelCSR(int size, double *A, double *value, int *colidx, int *rowptr)
{
	int tid = threadIdx.x;
	int idx = rowptr[tid], count = 0;
	for(int i = 0; i < size; i++)
	{
		if(A[tid*size + i] != 0)
		{
			value[idx + count] = A[tid*size + i];
			colidx[idx + count] = i;
			count ++;
		}
	}
}

int main()
{
	int size;
	double *A, *d_A, *value, *d_value;
	float time;
	int *colidx, *d_colidx, *rowptr, *d_rowptr, *d_nonzero;
	char filename[L];
	FILE *fp;

	printf("What's your file name?\n");
	scanf("%s", filename);

	printf("What is the size of the matrix A?\n");
	scanf("%d", &size);

	A = (double*) malloc (size*size*sizeof(*A));
	rowptr = (int*) malloc ((size+1)*sizeof(*rowptr));
	rowptr[0] = 0;

	fp = fopen(filename, "r");
	if(!fp)
	{
		printf("Fail to open the file!\n");
		return 0;
	}
	else
		printf("Success to open the file!\n\n");

	printf("Start to input the dense matrix...\n");

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			fscanf( fp, "%lf", &A[i*size + j]);
		}
	}
	printf("Done!\n");
	fclose(fp);
	printf("Close the file!\n");
	printf("====================================\n\n");

	printf("Start to convert the dense matrix to sparse matrix with CSR!\n");

	cudaEvent_t start, stop;

	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord(start, 0);

	cudaMalloc( (void**) &d_A, size*size*sizeof(double));
	cudaMalloc( (void**) &d_rowptr, (size+1)*sizeof(int));
	cudaMalloc( (void**) &d_nonzero, size*sizeof(int));

	cudaMemcpy( d_A, A, size*size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( d_rowptr, rowptr, (size+1)*sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(N);
  	dim3 grid((int)(size/N + 1));

  	kernelcount_nonz<<< grid, block>>>(size, d_A, d_nonzero, d_rowptr);

  	cudaMemcpy( rowptr, d_rowptr, (size+1)*sizeof(int), cudaMemcpyDeviceToHost);

  	cudaMalloc( (void**) &d_value, rowptr[size]*sizeof(double));
	cudaMalloc( (void**) &d_colidx, rowptr[size]*sizeof(int));

  	kernelCSR<<< grid, block>>>(size, d_A, d_value, d_colidx, d_rowptr);

  	value = (double*) malloc (rowptr[size]*sizeof(*value));
  	colidx = (int*) malloc (rowptr[size]*sizeof(*colidx));

  	cudaMemcpy( value, d_value, rowptr[size]*sizeof(double), cudaMemcpyDeviceToHost);
  	cudaMemcpy( colidx, d_colidx, rowptr[size]*sizeof(int), cudaMemcpyDeviceToHost);

  	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf( "Global memory processing time: %f (ms)\n" , time);

 	cudaFree(d_A);
  	cudaFree(d_value);
  	cudaFree(d_colidx);
  	cudaFree(d_rowptr);
  	cudaFree(d_nonzero);
  	printf("Successfully free the cuda memory\n");

	printf("====================================\n\n");

	char file[] = "CSR_gpu.txt";

	fp = fopen(file, "w");
	if(!fp)
	{
		printf("Fail to open the file!\n");
		return 0;
	}
	else
		printf("Start to write the file!\n");

	for(int i = 0; i < rowptr[size]; i++)
	{
		if(i <= rowptr[size-1])
		{
			fprintf( fp,"%f,", value[i]);
		}
		else
		{
			fprintf( fp, "%f\n", value[i]);
		}
	}

	for(int i = 0; i < rowptr[size]; i++)
	{
		if(i <= rowptr[size-1])
		{
			fprintf( fp, "%d,", colidx[i]);
		}
		else
		{
			fprintf( fp, "%d\n", colidx[i]);
		}
	}

	for(int i = 0; i < size + 1; i++)
	{
		if(i <= size)
		{
			fprintf( fp, "%d,", rowptr[i]);
		}
		else
		{
			fprintf( fp, "%d\n", rowptr[i]);
		}
	}

	fclose(fp);
	printf("Close the file!\n");

	free(A);
	free(value);
	free(colidx);
	free(rowptr);
	printf("Successfully free the memory\n\n");

	return 0;
}