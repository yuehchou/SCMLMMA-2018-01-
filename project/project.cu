#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#define L 100
#define N 128

double RandomNumber () { return ((double)rand()/RAND_MAX); }

void randmatrix (int size, double *A, double *copy_A);

int input_data(char *file, int size, double *A, double *copy_A);

void direct_transpose(int size, double *A);

void block_transpose(int block_n, int size, double *A);

void diag_block(int size, int block_row, int block_col, int block_size, double *A);

void nondiag_block(int size, int block_row, int block_col, int block_size, double *A);

double MP_block_transpose(int block_n, int size, double *A);

double CUDA_block_transpose(int block_n, int size, double *A);

__global__ void kernel_trans( int block_n, int block_size, int size, double *d_A);

double error_norm(int size, double *A, double *copy_A);

int main()
{
	int size, choice;
	double *A, *copy_A;
	double error, time;
	char ans;

	printf("Do you want to input data from your file?\n");
	printf("('y' for yes, 'n' for No)\n");
	scanf("%c", &ans);

	if (ans == 'y' || ans == 'Y')
	{
		char filename[L];
		int check;

		printf("What's your file name?\n");
		scanf("%s", filename);

		printf("What is the size of the matrix A?\n");
		scanf("%d", &size);

		A = (double*) malloc (size*size*sizeof(*A));
		copy_A = (double*) malloc (size*size*sizeof(*copy_A));

		check = input_data(filename, size, A, copy_A);

		if(check == 0)
			return 0;
	}

	else if (ans == 'n' || ans == 'N')
	{
		printf("Okay, we will construct the random matrix.\n");

		printf("What is the size of the matrix A?\n");
		scanf("%d", &size);

		A = (double*) malloc (size*size*sizeof(*A));
		copy_A = (double*) malloc (size*size*sizeof(*copy_A));

		printf("\nStart to construct the rand matrix A...\n\n");
		randmatrix (size, A, copy_A);
		printf("Done!!\n");
		printf("====================================\n\n");
	}

	else
	{
		printf("Wrong instruction!!\n");
		return 0;
	}

	printf("What's the way you want to choose to complete the transpose of this matrix A?\n");
	printf("1. Dierctly construct the transpose of A.\n");
	printf("2. Cut the several blocks in the matrix A to complete its transpose.\n");
	printf("3. Do 2. by using OpenMP\n");
	printf("4. Do 2. by using CUDA\n");

	scanf("%d", &choice);

	if (choice == 1)
	{
		double start, end;

		printf("\nStart to complete the transpose of A...\n\n");

		start = clock();
		direct_transpose( size, A);
		end = clock();

		printf("Done!!\n");
		printf("====================================\n\n");

		direct_transpose( size, A);
		error = error_norm(size, A, copy_A);
		printf("Error norm: %lf \n", error);
		time = (double)(end - start)/CLOCKS_PER_SEC;
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 2)
	{
		double start, end;
		int block_n;

		printf("\nStart to complete the transpose of A...\n\n");
		printf("We will cut this matrix A to nxn blocks.\n");
		printf("So, what is the integer n that you want to set?\n");
		scanf("%d", &block_n);

		start = clock();
		block_transpose(block_n, size, A);
		end = clock();

		printf("Done!!\n");
		printf("====================================\n\n");

		block_transpose(block_n, size, A);
		error = error_norm(size, A, copy_A);
		time = (double)(end - start)/CLOCKS_PER_SEC;
		printf("Error norm: %lf \n", error);
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 3)
	{
		int block_n;

		printf("\nStart to complete the transpose of A...\n\n");
		printf("We will cut this matrix A to nxn blocks.\n");
		printf("So, what is the integer n that you want to set?\n");
		scanf("%d", &block_n);

		time = MP_block_transpose( block_n, size, A);

		printf("Done!!\n");
		printf("====================================\n\n");

		MP_block_transpose( block_n, size, A);
		error = error_norm(size, A, copy_A);
		printf("Error norm: %lf \n", error);
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 4)
	{
		int block_n;

		printf("\nStart to complete the transpose of A...\n\n");
		printf("We will cut this matrix A to nxn blocks.\n");
		printf("So, what is the integer n that you want to set?\n");
		scanf("%d", &block_n);

		time = CUDA_block_transpose( block_n, size, A);

		printf("Done!!\n");
		printf("====================================\n\n");

		CUDA_block_transpose( block_n, size, A);
		error = error_norm(size, A, copy_A);
		printf("Error norm: %lf \n", error);
		printf( "Global memory processing time: %lf (ms)\n" , time);
	}

	else
	{
		printf("Wrong instruction!!\n");
		return 0;
	}

	return 0;
}

void randmatrix (int size, double *A, double *copy_A)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			A[i*size + j] = RandomNumber();
		}
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			copy_A[i*size + j] = A[i*size + j];
		}
	}
}

int input_data(char *file, int size, double *A, double *copy_A)
{
	FILE *fp;

	fp = fopen(file, "r");
	if(!fp)
	{
		printf("Fail to open the file!\n");
		return 0;
	}
	else
		printf("Success to open the file!\n\n");

	printf("\nStart to input the dense matrix...\n\n");

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			fscanf( fp, "%lf", &A[i*size + j]);
			copy_A[i*size + j] = A[i*size + j];
		}
	}

	printf("Done!\n");
	fclose(fp);
	printf("Close the file!\n");
	printf("====================================\n\n");

	return 1;
}

void direct_transpose(int size, double *A)
{
	double temp;

	for (int i = 0; i < size; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			temp = A[j*size + i];
			A[j*size + i] = A[i*size + j];
			A[i*size + j] = temp;
		}
	}
}

void block_transpose(int block_n, int size, double *A)
{
	int block_size;

	block_size = (int)(size / block_n + 1);

	for (int i = 0; i < block_n; i++)
	{
		diag_block( size, i, i, block_size, A);
		for (int j = i + 1; j < block_n; j++)
		{
			nondiag_block(size, i, j, block_size, A);
		}
	}
}

void diag_block(int size, int block_row, int block_col, int block_size, double *A)
{
	int rstart = block_row*block_size, rend = (block_row + 1)*block_size;
	int cend = (block_col + 1)*block_size;

	if(rend > size)
		rend = size;
	if(cend > size)
		cend = size;

	double temp;

	for(int i = rstart; i < rend; i++)
	{
		#pragma unroll(4)
		for(int j = i + 1; j < cend; j++)
		{
			temp = A[j*size + i];
			A[j*size + i] = A[i*size + j];
			A[i*size + j] = temp;
		}
	}
}

void nondiag_block(int size, int block_row, int block_col, int block_size, double *A)
{
	int rstart = block_row*block_size, rend = (block_row + 1)*block_size;
	int cstart = block_col*block_size, cend = (block_col + 1)*block_size;

	if(rend > size)
		rend = size;
	if(cend > size)
		cend = size;

	double temp;

	for(int i = rstart; i < rend; i++)
	{
		#pragma unroll(4)
		for(int j = cstart; j < cend; j++)
		{
			temp = A[j*size + i];
			A[j*size + i] = A[i*size + j];
			A[i*size + j] = temp;
		}
	}
}

double MP_block_transpose( int block_n, int size, double *A)
{
	int block_size;
	struct timespec start, end;

	block_size = (int)(size / block_n + 1);

	omp_set_num_threads(24);

	clock_gettime(CLOCK_REALTIME, &start);

	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < block_n; i++)
		{
			diag_block( size, i, i, block_size, A);
			for (int j = i + 1; j < block_n; j++)
			{
				nondiag_block(size, i, j, block_size, A);
			}
		}
	}
	clock_gettime(CLOCK_REALTIME, &end);

	return (end.tv_sec - start.tv_sec + (double)(end.tv_nsec - start.tv_nsec)/1e9);
}

double CUDA_block_transpose(int block_n, int size, double *A)
{
	int block_size;
	float time;
	double *d_A;

	block_size = (int)(size / block_n + 1);

	cudaEvent_t start, stop;

	cudaMalloc( (void**) &d_A, size*size*sizeof(double));

	cudaMemcpy( d_A, A, size*size*sizeof(double), cudaMemcpyHostToDevice);

	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord(start, 0);

	dim3 block(N,N);
	dim3 grid( (int)(block_n/N + 1), (int)(block_n/N + 1));
	kernel_trans<<<grid,block>>>( block_n, block_size, size, d_A);

  	cudaMemcpy( A, d_A, size*size*sizeof(double), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	cudaFree(d_A);

	return (double)time;
}

__global__ void kernel_trans( int block_n, int block_size, int size, double *d_A)
{
	int block_row = blockIdx.y * blockDim.y + threadIdx.y, block_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (block_row < block_n && block_col < block_n)
	{
		int rstart = block_row*block_size, rend = (block_row + 1)*block_size;
		int cstart = block_col*block_size, cend = (block_col + 1)*block_size;
		double temp;

		if(rend > size)
			rend = size;
		if(cend > size)
			cend = size;

		if(block_row == block_col)
		{
			for(int i = rstart; i < rend; i++)
			{
				for(int j = i + 1; j < cend; j++)
				{
					temp = d_A[j*size + i];
					d_A[j*size + i] = d_A[i*size + j];
					d_A[i*size + j] = temp;
				}
			}
		}

		else if(block_row < block_col)
		{
			for(int i = rstart; i < rend; i++)
			{
				for(int j = cstart; j < cend; j++)
				{
					temp = d_A[j*size + i];
					d_A[j*size + i] = d_A[i*size + j];
					d_A[i*size + j] = temp;
				}
			}
		}
	}
}

double error_norm(int size, double *A, double *copy_A)
{
	double temp, error = 0.0;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			temp = A[i*size + j] - copy_A[i*size + j];
			temp = temp*temp;
			error += temp;
		}
	}

	return sqrt(error);
}