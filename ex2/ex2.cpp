#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>
#define N 25
using namespace std;

double RandomNumber () { return ((double)rand()/RAND_MAX); }

void Poisson1D (int size, vector<double> &A);

void degmv(int size, vector<double> &A, vector<double> &correctx, vector<double> &b);

void LU(int size, vector<double> &A);

void pivot_LU(int size, vector<double> &A, vector<double> &pivot);

void solve_LU(int size, vector<double> &A, vector<double> &b, vector<double> &x);

void solve_pivot_LU(int size, vector<double> &A, vector<double> &b, vector<double> &x, vector<double> &pivot);

void error_norm(int size, vector<double> &correctx, vector<double> &x, double &error);

void output_resluts(char *file, vector<double> &data, int size);

int main()
{
	int size, t;
	double start, end;
	char ans;

	cout<<"What is the size of your matrix A?\n";
	cin>>size;
	t = size/N + 1;

	vector<double> error(t + 1), time(t + 1), gflops(t + 1);

	cout<<"Do you want to enhance the LU factorization with pivoting?\n";
	cin>>ans;

	if (ans == 'y'|| ans == 'Y')
	{
		cout<<"Ok, let's start to do the LU factorization with pivoting...\n";

		for(int i = 0; i <= t; i++)
		{
			int s = i*N;

			if( i == 0)
			{
				vector<double> A(1), correctx(1), x(1), b(1), pivot(1);

				generate( A.begin(), A.end(), RandomNumber);
				//Poisson1D( 1, A);
				generate( correctx.begin(), correctx.end(), RandomNumber);
				
				for(int j = 0; j < s; j++)
				{
					pivot[j] = j;
				}

				degmv(1, A, correctx, b);

				start = clock();
				pivot_LU(1, A, pivot);
				end = clock();

				solve_pivot_LU( 1, A, b, x, pivot);

				error_norm(1, correctx, x, error[i]);

				time[i] = (double)(end - start)/CLOCKS_PER_SEC;
				gflops[i] = 2 / 3 / time[i] / 1e9;
			}
			else if(i == t && size%N != 0)
			{
				vector<double> A(size*size), correctx(size), x(size), b(size), pivot(size);

				generate( A.begin(), A.end(), RandomNumber);
				//Poisson1D( size, A);
				generate( correctx.begin(), correctx.end(), RandomNumber);
				//A is stored by column major

				for(int j = 0; j < size; j++)
				{
					pivot[j] = j;
				}

				degmv(size, A, correctx, b);

				start = clock();
				pivot_LU(size, A, pivot);
				end = clock();

				solve_pivot_LU( size, A, b, x, pivot);

				error_norm(size, correctx, x, error[i]);

				time[i] = (double)(end - start)/CLOCKS_PER_SEC;
				gflops[i] = 2*(double)size*size*size / 3 / time[i] / 1e9;
			}
			else
			{
				vector<double> A(s*s), correctx(s), x(s), b(s), pivot(s);

				generate( A.begin(), A.end(), RandomNumber);
				//Poisson1D( s, A);
				generate( correctx.begin(), correctx.end(), RandomNumber);
				
				for(int j = 0; j < s; j++)
				{
					pivot[j] = j;
				}

				degmv(s, A, correctx, b);

				start = clock();
				pivot_LU(s, A, pivot);
				end = clock();

				solve_pivot_LU( s, A, b, x, pivot);

				error_norm(s, correctx, x, error[i]);

				time[i] = (double)(end - start)/CLOCKS_PER_SEC;
				gflops[i] = 2*(double)s*s*s / 3 / time[i] / 1e9;	
			}
		}

		cout<<"Done!\n";

		char file[] = "error_pivot.txt", fileT[] = "time_pivot.txt", fileGFS[] = "GFlops_pivot.txt";

		output_resluts(file, error, t);

		output_resluts(fileT, time, t);

		output_resluts(fileGFS, gflops, t);	
	}
	
	else if(ans == 'n'|| ans == 'N')
	{
		cout<<"Ok, let's start to do the LU factorization...\n";

		for(int i = 0; i <= t; i++)
		{
			int s = i*N;
			if(i == 0)
			{
				vector<double> A(1), correctx(1), x(1), b(1);

				generate( A.begin(), A.end(), RandomNumber);
				//Poisson1D( 1, A);
				generate( correctx.begin(), correctx.end(), RandomNumber);

				degmv(1, A, correctx, b);

				start = clock();
				LU(1, A);
				end = clock();

				solve_LU( 1, A, b, x);

				error_norm(1, correctx, x, error[i]);

				time[i] = (double)(end - start)/CLOCKS_PER_SEC;
				gflops[i] = 2 / 3 / time[i] / 1e9;
			}
			else if(i == t && size%N != 0)
			{
				vector<double> A(size*size), correctx(size), x(size), b(size);

				generate( A.begin(), A.end(), RandomNumber);
				//Poisson1D( size, A);
				generate( correctx.begin(), correctx.end(), RandomNumber);

				degmv(size, A, correctx, b);

				start = clock();
				LU(size, A);
				end = clock();

				solve_LU( size, A, b, x);

				error_norm(size, correctx, x, error[i]);

				time[i] = (double)(end - start)/CLOCKS_PER_SEC;
				gflops[i] = 2*(double)size*size*size / 3 / time[i] / 1e9;
			}
			else
			{
				vector<double> A(s*s), correctx(s), x(s), b(s);

				generate( A.begin(), A.end(), RandomNumber);
				//Poisson1D( s, A);
				generate( correctx.begin(), correctx.end(), RandomNumber);

				degmv(s, A, correctx, b);

				start = clock();
				LU(s, A);
				end = clock();

				solve_LU( s, A, b, x);

				error_norm(s, correctx, x, error[i]);

				time[i] = (double)(end - start)/CLOCKS_PER_SEC;
				gflops[i] = 2*(double)s*s*s / 3 / time[i] / 1e9;
			}
		}
		cout<<"Done!\n";

		char file[] = "error.txt", fileT[] = "time.txt", fileGFS[] = "GFlops.txt";

		output_resluts(file, error, t);

		output_resluts(fileT, time, t);

		output_resluts(fileGFS, gflops, t);
	}

	else
	{
		cout<<"Wrong instruction!!\n";
		return 0;
	}

	return 0;
}

void Poisson1D (int size, vector<double> &A)
{
	if(size == 1)
	{
		A[size -1] = 1;
	}
	else
	{
		for(int i = 0; i < size; i++)
		{
			for(int j = 0; j < size; j++)
			{
				if(i == j)
					A[i*size + j] = 2;
				else if (j == i - 1 || j == i + 1)
					A[i*size + j] = -1;
				else
					A[i*size + j] = 0;
			}
		}
	}
}

void degmv(int size, vector<double> &A, vector<double> &correctx, vector<double> &b)
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

void LU(int size, vector<double> &A)
{
	double temp;

	for(int i = 0; i < size; i++)
	{
		temp = 1.0/A[i*size + i];
		for(int j = i + 1; j < size; j++)
		{
			A[i*size + j] *= temp;
		}

		for(int j = i + 1; j < size; j++)
		{
			for(int k = i + 1; k < size; k++)
			{
				A[j*size + k] -= A[j*size + i] * A[i*size + k];
			}
		}
	}
}

void pivot_LU(int size, vector<double> &A, vector<double> &pivot)
{
	int itemp, idx;
	double temp, max;

	for(int i = 0; i < size; i++)
	{
		max = A[i*size + i];
		idx = i;

		for(int j = i; j < size; j++)
		{
			if(A[j*size + i] > max)
			{
				max = A[j*size + i];
				idx = j;
			}
		}

		if(idx != i)
		{
			itemp = pivot[i];
			pivot[i] = pivot[idx];
			pivot[idx] = itemp;

			vector<double> t(size);

			copy( &A[i*size], &A[(i+1)*size], t.begin());
			copy( &A[idx*size], &A[(idx + 1)*size], &A[i*size]);
			copy( t.begin(), t.end(), &A[idx*size]);
		}

		temp = 1.0/A[i*size + i];

		for(int j = i + 1; j < size; j++)
		{
			A[i*size + j] *= temp;
		}

		for(int j = i + 1; j < size; j++)
		{
			for(int k = i + 1; k < size; k++)
			{
				A[j*size + k] -= A[j*size + i] * A[i*size + k];
			}
		}
	}
}

void solve_LU(int size, vector<double> &A, vector<double> &b, vector<double> &x)
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

void solve_pivot_LU(int size, vector<double> &A, vector<double> &b, vector<double> &x, vector<double> &pivot)
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