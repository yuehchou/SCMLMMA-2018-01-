#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mkl.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#define L 100
#define N 32

double RandomNumber () { return ((double)rand()/RAND_MAX); }

void randmatrix (int row, int col, double *A, double *copy_A);

int input_data(char *file, int row, int col, double *A, double *copy_A);

void direct_transpose(int row, int col, double *A, double *trA);

double MP_direct_transpose(int row, int col, double *A, double *trA);

double CUDA_direct_transpose(int row, int col, double *A, double *trA);

__global__ void kernel_rtrans(int row, int col, double *d_A, double *d_trA);

__global__ void kernel_ctrans(int row, int col, double *d_A, double *d_trA);

void block_transpose(int block_m, int block_n, int row, int col, double *A, double *trA);

double MP_block_transpose(int block_m, int block_n, int row, int col, double *A, double *trA);

double CUDA_block_transpose(int block_m, int block_n, int row, int col, double *A, double *trA);

__global__ void kernel_block_trans(int block_m, int block_n, int block_rsize, int block_csize, int row, int col, double *d_A, double *d_trA);

double error_norm(int row, int col, double *A, double *copy_A);

int main()
{
	int row, col, choice;
	double *A, *trA, *copy_A;
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

		printf("What is the number of the row in the matrix A?\n");
		scanf("%d", &row);

		printf("What is the number of the column in the matrix A?\n");
		scanf("%d", &col);

		A = (double*) malloc (row*col*sizeof(*A));
		trA = (double*) malloc (row*col*sizeof(*trA));
		copy_A = (double*) malloc (row*col*sizeof(*copy_A));

		check = input_data(filename, row, col, A, copy_A);

		if(check == 0)
			return 0;
	}

	else if (ans == 'n' || ans == 'N')
	{
		printf("Okay, we will construct the random matrix.\n");

		printf("What is the number of the row in the matrix A?\n");
		scanf("%d", &row);

		printf("What is the number of the column in the matrix A?\n");
		scanf("%d", &col);

		A = (double*) malloc (row*col*sizeof(*A));
		trA = (double*) malloc (row*col*sizeof(*trA));
		copy_A = (double*) malloc (row*col*sizeof(*copy_A));

		printf("\nStart to construct the rand matrix A...\n\n");
		randmatrix ( row, col, A, copy_A);
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
	printf("2. Do 1. by using OpenMP\n");
	printf("3. Do 1. by using CUDA\n");
	printf("4. Cut the several blocks in the matrix A to complete its transpose.\n");
	printf("5. Do 4. by using OpenMP\n");
	printf("6. Do 4. by using CUDA\n");
	printf("7. Using MKL.\n");

	scanf("%d", &choice);

	if (choice == 1)
	{
		double start, end;

		printf("\nStart to complete the transpose of A...\n\n");

		start = clock();
		direct_transpose( row, col, A, trA);
		end = clock();

		printf("Done!!\n");
		printf("====================================\n\n");

		direct_transpose( col, row, trA, A);
		error = error_norm(row, col, A, copy_A);
		printf("Error norm: %lf \n", error);
		time = (double)(end - start)/CLOCKS_PER_SEC;
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 2)
	{
		printf("\nStart to complete the transpose of A...\n\n");

		time = MP_direct_transpose( row, col, A, trA);

		printf("Done!!\n");
		printf("====================================\n\n");

		MP_direct_transpose( col, row, trA, A);
		error = error_norm( row, col, A, copy_A);
		printf("Error norm: %lf \n", error);
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 3)
	{
		printf("\nStart to complete the transpose of A...\n\n");

		time = CUDA_direct_transpose( row, col, A, trA);

		printf("Done!!\n");
		printf("====================================\n\n");

		CUDA_direct_transpose( col, row, trA, A);
		error = error_norm( row, col, A, copy_A);
		printf("Error norm: %lf \n", error);
		printf( "Global memory processing time: %lf (ms)\n" , time);
	}

	else if (choice == 4)
	{
		double start, end;
		int block_m, block_n;

		printf("\nStart to complete the transpose of A...\n\n");

		printf("We will cut this matrix A to mxn blocks.\n");
		printf("So, what is the integer m that you want to set?\n");
		scanf("%d", &block_m);
		printf("And, what is the integer n that you want to set?\n");
		scanf("%d", &block_n);

		start = clock();
		block_transpose( block_m, block_n, row, col, A, trA);
		end = clock();

		printf("Done!!\n");
		printf("====================================\n\n");

		block_transpose( block_n, block_m, col, row, trA, A);
		error = error_norm(row, col, A, copy_A);
		time = (double)(end - start)/CLOCKS_PER_SEC;
		printf("Error norm: %lf \n", error);
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 5)
	{
		int block_m, block_n;
		printf("\nStart to complete the transpose of A...\n\n");

		printf("We will cut this matrix A to mxn blocks.\n");
		printf("So, what is the integer m that you want to set?\n");
		scanf("%d", &block_m);
		printf("And, what is the integer n that you want to set?\n");
		scanf("%d", &block_n);

		time = MP_block_transpose( block_m, block_n, row, col, A, trA);

		printf("Done!!\n");
		printf("====================================\n\n");

		MP_block_transpose( block_n, block_m, col, row, trA, A);
		error = error_norm( row, col, A, copy_A);
		printf("Error norm: %lf \n", error);
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 6)
	{
		int block_m, block_n;
		printf("\nStart to complete the transpose of A...\n\n");

		printf("We will cut this matrix A to mxn blocks.\n");
		printf("So, what is the integer m that you want to set?\n");
		scanf("%d", &block_m);
		printf("And, what is the integer n that you want to set?\n");
		scanf("%d", &block_n);

		time = CUDA_block_transpose( block_m, block_n, row, col, A, trA);

		printf("Done!!\n");
		printf("====================================\n\n");

		CUDA_block_transpose( block_n, block_m, col, row, trA, A);
		error = error_norm( row, col, A, copy_A);
		printf("Error norm: %lf \n", error);
		printf( "Global memory processing time: %lf (ms)\n" , time);
	}

	else if (choice == 7)
	{
		struct timespec start, end;
		printf("\nStart to complete the transpose of A...\n\n");

		clock_gettime(CLOCK_REALTIME, &start);
		mkl_domatcopy('r', 't', row, col, 1, A, col, trA, row);
		clock_gettime(CLOCK_REALTIME, &end);

		printf("Done!!\n");
		printf("====================================\n\n");
		time = (end.tv_sec - start.tv_sec + (double)(end.tv_nsec - start.tv_nsec)/1e9);
		printf("Cost time: %lf \n\n", time);
	}

	else if (choice == 8)
	{
		int block_m, block_n;
		time = 0.0;
		printf("\nStart to complete the transpose of A...\n\n");
		printf("We will cut this matrix A to mxn blocks.\n");
		printf("So, what is the integer m that you want to set?\n");
		scanf("%d", &block_m);
		printf("And, what is the integer n that you want to set?\n");
		scanf("%d", &block_n);

		for(int i = 0; i < 20; i++)
		{
			if ( i > 4 && i < 15 )
				time += MP_block_transpose( block_m, block_n, row, col, A, trA);
			else
				MP_block_transpose( block_m, block_n, row, col, A, trA);
			
			MP_block_transpose( block_n, block_m, col, row, trA, A);
		}
		
		printf("Done!!\n");
		printf("====================================\n\n");
		printf("Average Cost time: %lf \n\n", time/10);
	}

	else if (choice == 9)
	{
		struct timespec start, end;
		time = 0.0;

		for(int i = 0; i < 20; i++)
		{
			clock_gettime(CLOCK_REALTIME, &start);
			mkl_domatcopy('r', 't', row, col, 1, A, col, trA, row);
			clock_gettime(CLOCK_REALTIME, &end);

			mkl_domatcopy('r', 't', col, row, 1, trA, row, A, col);

			if ( i > 4 && i < 15 )
				time += (end.tv_sec - start.tv_sec + (double)(end.tv_nsec - start.tv_nsec)/1e9);
		}

		printf("Done!!\n");
		printf("====================================\n\n");
		printf("Average cost time: %lf \n\n", time/10);
	}

	else
	{
		printf("Wrong instruction!!\n");
		return 0;
	}

	return 0;
}

void randmatrix (int row, int col, double *A, double *copy_A)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			A[i*col + j] = RandomNumber();
			copy_A[i*col + j] = A[i*col + j];
		}
	}
}

int input_data(char *file, int row, int col, double *A, double *copy_A)
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

	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			fscanf( fp, "%lf", &A[i*col + j]);
			copy_A[i*col + j] = A[i*col + j];
		}
	}

	printf("Done!\n");
	fclose(fp);
	printf("Close the file!\n");
	printf("====================================\n\n");

	return 1;
}

void direct_transpose(int row, int col, double *A, double *trA)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			trA[j*row + i] = A[i*col + j];
		}
	}
}

double MP_direct_transpose(int row, int col, double *A, double *trA)
{
	struct timespec start, end;

	omp_set_num_threads(24);

	clock_gettime(CLOCK_REALTIME, &start);

	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < row; i++)
		{
			#pragma unroll(4)
			for (int j = 0; j < col; j++)
			{
				trA[j*row + i] = A[i*col + j];
			}
		}
	}

	clock_gettime(CLOCK_REALTIME, &end);

	return (end.tv_sec - start.tv_sec + (double)(end.tv_nsec - start.tv_nsec)/1e9);
}

double CUDA_direct_transpose(int row, int col, double *A, double *trA)
{
	float time;
	double *d_A, *d_trA;
	cudaEvent_t start, stop;

	cudaMalloc( (void**) &d_A, row*col*sizeof(double));
	cudaMalloc( (void**) &d_trA, row*col*sizeof(double));

	cudaMemcpy( d_A, A, row*col*sizeof(double), cudaMemcpyHostToDevice);

	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord(start, 0);

	if( row >= col)
	{
		dim3 block(1024);
		dim3 grid((int)(row/1024 + 1));
		kernel_rtrans<<<grid, block>>>( row, col, d_A, d_trA);
	}
	else
	{
		dim3 block(1024);
		dim3 grid((int)(col/1024 + 1));
		kernel_ctrans<<<grid, block>>>( row, col, d_A, d_trA);
	}

	cudaMemcpy( trA, d_trA, row*col*sizeof(double), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	cudaFree(d_A);
	cudaFree(d_trA);

	return (double)time;
}

__global__ void kernel_rtrans(int row, int col, double *d_A, double *d_trA)
{
	int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x;

	if (bid*bdim + tid < row)
	{
		for (int i = 0; i < col; i++)
		{
			d_trA[ i*row + bid*bdim + tid] = d_A[(bid*bdim + tid)*col + i];
		}
	}
}

__global__ void kernel_ctrans(int row, int col, double *d_A, double *d_trA)
{
	int tid = threadIdx.x, bid = blockIdx.x, bdim = blockDim.x;

	if (bid*bdim + tid < col)
	{
		for (int i = 0; i < row; i++)
		{
			d_trA[ (bid*bdim + tid)*row + i] = d_A[i*col + bid*bdim + tid];
		}
	}
}

void block_transpose( int block_m, int block_n, int row, int col, double *A, double *trA)
{
	int block_rsize, block_csize;

	block_rsize = (int)(row / block_m + 1);
	block_csize = (int)(col / block_n + 1);

	for (int i = 0; i < block_m; i++)
	{
		for (int j = 0; j < block_n; j++)
		{
			int rstart = i*block_rsize, rend = (i + 1)*block_rsize;
			int cstart = j*block_csize, cend = (j + 1)*block_csize;

			if(rend > row)
				rend = row;
			if(cend > col)
				cend = col;

			for(int k = rstart; k < rend; k++)
			{
				#pragma unroll(4)
				for(int l = cstart; l < cend; l++)
				{
					trA[l*row + k] = A[k*col + l];
				}
			}
		}
	}
}

double MP_block_transpose(int block_m, int block_n, int row, int col, double *A, double *trA)
{
	struct timespec start, end;
	int block_rsize, block_csize;

	block_rsize = (int)(row / block_m + 1);
	block_csize = (int)(col / block_n + 1);

	omp_set_num_threads(24);

	clock_gettime(CLOCK_REALTIME, &start);

	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < block_m; i++)
		{
			for (int j = 0; j < block_n; j++)
			{
				int rstart = i*block_rsize, rend = (i + 1)*block_rsize;
				int cstart = j*block_csize, cend = (j + 1)*block_csize;

				if(rend > row)
					rend = row;
				if(cend > col)
					cend = col;

				for(int k = rstart; k < rend; k++)
				{
					#pragma unroll(4)
					for(int l = cstart; l < cend; l++)
					{
						trA[l*row + k] = A[k*col + l];
					}
				}
			}
		}
	}

	clock_gettime(CLOCK_REALTIME, &end);

	return (end.tv_sec - start.tv_sec + (double)(end.tv_nsec - start.tv_nsec)/1e9);
}

double CUDA_block_transpose(int block_m, int block_n, int row, int col, double *A, double *trA)
{
	float time;
	double *d_A, *d_trA;
	int block_rsize, block_csize;
	cudaEvent_t start, stop;

	block_rsize = (int)(row / block_m + 1);
	block_csize = (int)(col / block_n + 1);

	cudaMalloc( (void**) &d_A, row*col*sizeof(double));
	cudaMalloc( (void**) &d_trA, row*col*sizeof(double));

	cudaMemcpy( d_A, A, row*col*sizeof(double), cudaMemcpyHostToDevice);

	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord(start, 0);

	dim3 block( N, N);
	dim3 grid( (int)(block_m/N + 1), (int)(block_n/N + 1));

	//printf("%d\n", grid.x);
	kernel_block_trans <<< grid, block>>>( block_m, block_n, block_rsize, block_csize, row, col, d_A, d_trA);
	//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	// assert(cudaGetLastError() == cudaSuccess);

  	cudaMemcpy( trA, d_trA, row*col*sizeof(double), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	cudaFree(d_A);
	cudaFree(d_trA);

	return (double)time;
}

__global__ void kernel_block_trans( int block_m, int block_n, int block_rsize, int block_csize, int row, int col, double *d_A, double *d_trA)
{
	int block_row = blockIdx.x * blockDim.x + threadIdx.x, block_col = blockIdx.y * blockDim.y + threadIdx.y;

	if (block_row < block_m && block_col < block_n)
	{
		int rstart = block_row*block_rsize, rend = (block_row + 1)*block_rsize;
		int cstart = block_col*block_csize, cend = (block_col + 1)*block_csize;

		if(rend > row)
			rend = row;
		if(cend > col)
			cend = col;

		for(int i = rstart; i < rend; i++)
		{
			#pragma unroll(4)
			for(int j = cstart; j < cend; j++)
			{
				d_trA[j*row + i] = d_A[i*col + j];
			}
		}
	}
}

double error_norm(int row, int col, double *A, double *copy_A)
{
	double temp, error = 0.0;

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			temp = A[i*col + j] - copy_A[i*col + j];
			temp = temp*temp;
			error += temp;
		}
	}

	return sqrt(error);
}