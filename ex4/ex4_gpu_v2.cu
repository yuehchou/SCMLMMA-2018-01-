#include <stdlib.h>
#include <stdio.h>
#define L 100

__global__ void kernelcount_nonz(int size, double *A, int *nonz, int *rowptr)
{
	int tid = threadIdx.x, count = 0;
	for(int i = 0; i < size; i++)
	{
		if(A[tid*size + i] != 0.0)
		{
			count ++;
		}
	}
	nonz[tid] = count;

	__syncthreads();

	count = 0;
	for(int i = 0; i <= tid; i++)
	{
		count += nonz[i];
	}
	rowptr[tid + 1] = count;
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
	char ans;
	FILE *fp;

	printf("Do you want to input the data from the file?\n");
	scanf("%c", &ans);

	if(ans == 'y' || ans == 'Y')
	{
		char filename[L];

		printf("\nWhat's your file name?\n");
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
	}
	else
	{
		int choice;
		printf("Ok, let's start to construct the matrix stored in dense.\n");
		printf("1. 0-1 criss-crossing matrix.\n");
		printf("2. 1D Poisson differential matrix.\n");
		printf("What's your choice?\n");
		scanf("%d", &choice);
		printf("What's the size of this matrix?\n");
		scanf("%d", &size);

		A = (double*) malloc (size*size*sizeof(*A));
		rowptr = (int*) malloc ((size+1)*sizeof(*rowptr));
		rowptr[0] = 0;

		if(choice == 1)
		{
			for(int i = 0; i < size; i++)
			{
				for(int j = 0; j < size; j++)
				{
					if((i + j)%2 == 0)
						A[i*size + j] = 0;
					else
						A[i*size + j] = 1;
				}
			}
		}
		else if (choice == 2)
		{
			for(int i = 0; i < size; i++)
			{
				for(int j = 0; j < size; j++)
				{
					A[i*size + j] = 0;
				}
			}

			for(int i = 0; i < size; i++)
			{
				A[i*size + i] = 2;
				A[i*size + i - 1] = -1;
				A[(i+1)*size + i] = -1;	
			}
			A[size*size - 1] = 2;
		}
		else
		{
			printf("No this choice!\n");
			return 0;
		}
		printf("Done\n");
	}

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

	dim3 block(size);
  	dim3 grid(1);

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
		if(i < rowptr[size])
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
		if(i < rowptr[size])
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
/*
	free(A);
	free(value);
	free(colidx);
	free(rowptr);
	printf("Successfully free the memory\n\n");
*/
	return 0;
}