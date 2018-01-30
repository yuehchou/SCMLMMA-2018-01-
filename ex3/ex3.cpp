#include <iostream>
#include <fstream>
#include <omp.h>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <cmath>
#include <algorithm>
#define N 50
#define D 50	
using namespace std;

double RandomNumber () { return ((double)rand()/RAND_MAX); }

void degmv(int size, double *A, vector<double> &correctx, vector<double> &b);

void diag_fact(int block_col, int size, double *A);

void row_update(int block_col, int block_row, int size, double *A);

void col_update(int block_col, int block_row, int size, double *A);

void trail_update(int block_col, int block_row, int block_travel, int size, double *A);

void block_LU(int num_blocks, int size, double *A);

void solve_LU(int size, double *A, vector<double> &b, vector<double> &x);

void error_norm(int size, vector<double> &correctx, vector<double> &x, double &error);

void output_resluts(char *file, vector<double> &data, int size);

int main()
{
	int size, t;
	struct timespec start, end;
	char ans;

	cout<<"What is the size of your matrix A?\n";
	cin>>size;
	t = size/N + 1;

	vector<double> error(t + 1), time(t + 1), gflops(t + 1);

	cout<<"Let's start to do the LU factorization...\n";

	for(int i = 0; i <= t; i++)
	{
		int s = i*N;
		if(i == 0)
		{
			vector<double> correctx(1), x(1), b(1);
			double *A;
			A = new double[1];

			generate( A, &A[1], RandomNumber);
			generate( correctx.begin(), correctx.end(), RandomNumber);

			degmv(1, A, correctx, b);

			clock_gettime(CLOCK_REALTIME, &start);
			block_LU(0, 1, A);
			clock_gettime(CLOCK_REALTIME, &end);

			solve_LU( 1, A, b, x);

			error_norm(1, correctx, x, error[i]);
			time[i] = end.tv_sec - start.tv_sec;
			time[i] += (double)(end.tv_nsec - start.tv_nsec)/ 1e9;
			gflops[i] = 2 / 3 / time[i] / 1e9;
		}
		else if(i == t && size%N != 0)
		{
			vector<double> correctx(size), x(size), b(size);
			double *A;
			A = new double[size*size];
			int num_blocks = (int)(size/D + 1);

			generate( A, &A[size*size], RandomNumber);
			generate( correctx.begin(), correctx.end(), RandomNumber);

			degmv(size, A, correctx, b);

			clock_gettime(CLOCK_REALTIME, &start);
			block_LU( num_blocks, size, A);
			clock_gettime(CLOCK_REALTIME, &end);

			solve_LU( size, A, b, x);

			error_norm(size, correctx, x, error[i]);

			time[i] = end.tv_sec - start.tv_sec;
			time[i] += (double)(end.tv_nsec - start.tv_nsec) / 1e9;
			gflops[i] = 2*(double)size*size*size / 3 / time[i] / 1e9;
		}
		else
		{
			vector<double> correctx(s), x(s), b(s);
			double *A;
			A = new double[s*s];
			int num_blocks = (int)(size/D + 1);

			generate( A, &A[s*s], RandomNumber);
			generate( correctx.begin(), correctx.end(), RandomNumber);

			degmv(s, A, correctx, b);

			clock_gettime(CLOCK_REALTIME, &start);
			block_LU( num_blocks, s, A);
			clock_gettime(CLOCK_REALTIME, &end);

			solve_LU( s, A, b, x);

			error_norm(s, correctx, x, error[i]);

			time[i] = end.tv_sec - start.tv_sec;
			time[i] += (double)(end.tv_nsec - start.tv_nsec) / 1e9;
			gflops[i] = 2*(double)s*s*s / 3 / time[i] / 1e9;
		}
	}
	cout<<"Done!\n";

	char file[] = "error.txt", fileT[] = "time.txt", fileGFS[] = "GFlops.txt";

	output_resluts(file, error, t);

	output_resluts(fileT, time, t);

	output_resluts(fileGFS, gflops, t);

	return 0;
}

void degmv(int size, double *A, vector<double> &correctx, vector<double> &b)
{
	double temp;

	for (int i = 0; i < size; i++)
	{	
		temp = 0.0;

		for (int j = 0; j < size; j++)
		{
			temp += A[j*size + i]*correctx[j];
		}

		b[i] = temp;
	}
}

void diag_fact(int block_col, int size, double *A)
{
	double temp;
	int start = block_col*N, end = (block_col+1)*N;

	if(end > size)
		end = size;

	for(int i = start; i < end; i++)
	{
		temp = 1.0/A[i*size + i];
		for(int j = i + 1; j < end; j++)
		{
			A[i*size + j] *= temp;
		}

		for(int j = i + 1; j < end; j++)
		{
			for(int k = i + 1; k < end; k++)
			{
				A[j*size + k] -= A[j*size + i] * A[i*size + k];
			}
		}
	}
}

void row_update(int block_col, int block_row, int size, double *A)
{
	double temp;
	int cstart = block_col*N, cend = (block_col+1)*N;
	int rstart = block_row*N, rend = (block_row+1)*N;

	if(cend > size)
		cend = size;

	if(rend > size)
		rend = size;

	for(int i = rstart; i < rend; i++)
	{
		for(int j = cstart; j < cend; j++)
		{
			temp = 0.0;
			for(int k = rstart; k < i; k++)
			{
				temp += A[k*size + i]*A[j*size + k];
			}
			A[j*size + i] -= temp;
		}
	}
}

void col_update(int block_col, int block_row, int size, double *A)
{
	double temp;
	int cstart = block_col*N, cend = (block_col+1)*N;
	int rstart = block_row*N, rend = (block_row+1)*N;

	if(cend > size)
		cend = size;

	if(rend > size)
		rend = size;

	for(int i = cstart; i < cend; i++)
	{
		for(int j = rstart; j < rend; j++)
		{
			temp = 0.0;
			for(int k = cstart; k < i; k++)
			{
				temp += A[k*size + j]*A[i*size + k];
			}
			A[i*size + j] = (A[i*size + j] - temp) / A[i*size + i];
		}
	}
}

void trail_update( int block_diag, int block_col, int block_row, int size, double *A)
{
	double temp;
	int dstart = block_diag*N, dend = (block_diag+1)*N;
	int cstart = block_col*N, cend = (block_col+1)*N;
	int rstart = block_row*N, rend = (block_row+1)*N;

	if(dend > size)
		dend = size;

	if(cend > size)
		cend = size;

	if(rend > size)
		rend = size;

	for(int i = rstart; i < rend; i++)
	{
		for(int j = cstart; j < cend; j++)
		{
			temp = 0.0;
			for(int k = dstart; k < dend; k++)
			{
				temp += A[k*size + i]*A[j*size + k];
			}
			A[j*size + i] -= temp;
		}
	}
}

void block_LU(int num_blocks, int size, double *A)
{
	omp_set_num_threads(24);

	#pragma omp parallel
	{
		#pragma omp single
		{
			for(int i = 0; i < num_blocks; i++)
			{
				int i_idx, j_idx, k_idx;
				i_idx = i*N;
				#pragma omp task depend( inout: A[i_idx*size + i_idx] )
				diag_fact( i, size, A);
				
				for(int j = i+1; j < num_blocks; j++)
				{
					j_idx = j*N;
					#pragma omp task depend( in: A[i_idx*size + i_idx] ) depend( inout: A[j_idx*size + i_idx] )
					row_update( j, i, size, A);
					#pragma omp task depend( in: A[i_idx*size + i_idx] ) depend( inout: A[i_idx*size + j_idx] )
					col_update( i, j, size, A);
				}

				for(int j = i+1; j < num_blocks; j++)
				{
					for(int k = i+1; k < num_blocks; k++)
					{
						j_idx = j*N;
						k_idx = k*N;
						#pragma omp task depend( in: A[i_idx*size + j_idx], A[k_idx*size + i_idx] ) depend( inout: A[k_idx*size + j_idx])
						trail_update( i, k, j, size, A);
					}
				}
			}
		}
	}
}

void solve_LU(int size, double *A, vector<double> &b, vector<double> &x)
{
	double temp;

	//solve Ly = b
	for(int i = 0; i < size; i++)
	{
		temp = 0.0;
		for(int j = 0; j < i; j++)
		{
			temp +=  A[j*size + i]*x[j];
		}
		x[i] = b[i] - temp;
	}

	//Ux = b'
	for(int i = size - 1; i >= 0; i--)
	{
		temp = 0.0;
		for(int j = size - 1; j > i; j--)
		{
			temp += A[j*size + i]*x[j];
		}
		x[i] = (x[i] - temp) / A[i*size + i];
	}
}

void solve_pivot_LU(int size, double *A, vector<double> &b, vector<double> &x, vector<double> &pivot)
{
	double temp;
	vector<double> tempx(size);

	//solve Ly = b
	for(int i = 0; i < size; i++)
	{
		temp = 0.0;
		for(int j = 0; j < i; j++)
		{
			temp +=  A[j*size + i]*x[j];
		}
		x[i] = b[i] - temp;
	}

	//Ux = b'
	for(int i = size - 1; i >= 0; i--)
	{
		temp = 0.0;
		for(int j = size - 1; j > i; j--)
		{
			temp += A[j*size + i]*x[j];
		}
		x[i] = (x[i] - temp) / A[i*size + i];
	}

	//return x
	for(int i = 0; i < size; i++)
	{
		tempx[pivot[i]] = x[i];
	}

	copy(tempx.begin(), tempx.end(), x.begin());
}

void error_norm(int size, vector<double> &correctx, vector<double> &x, double &error)
{
	double	norm = 0.0, temp;

	for(int i = 0; i < size; i++)
	{
		temp = correctx[i] - x[i];
		norm += temp*temp;
	}

	error =  sqrt(norm);
}

void output_resluts(char *file, vector<double> &data, int size)
{
	fstream fp;

	cout<<"=================================\n\n";

	fp.open(file, ios::out);
	if(!fp)
	{
		cout<<"Fail to open "<<file<<"!\n";
	}

	else
	{
		cout<<"Start to write "<<file<<"...\n";

		for(int i = 0; i < size; i++)
		{
			fp<<data[i]<<endl;
		}

		fp.close();
		cout<<"Close the file!\n";
	}
}