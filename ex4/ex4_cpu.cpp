#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#define L 100
using namespace std;

int main()
{
	int size, count_nonz, count;
	double start, end;
	double *A = NULL, *value = NULL;
	int *colidx = NULL, *rowptr = NULL;
	char filename[L];
	fstream fp;

	cout<<"What is your input file?\n";
	cin.getline(filename,L,'\n');
	cout<<"What is the size of the matrix A?\n";
	cin>>size;
	A = new double[size*size];
	value = new double[size*size];
	colidx = new int[size*size];
	rowptr = new int[size+1];

	fp.open(filename, ios::in);
	if(!fp)
	{
		cout<<"Fail to open "<<filename<<" !\n";
		return 0;
	}
	else
		cout<<"Success to open "<<filename<<" !\n\n";

	cout<<"Start to input the dense matrix...\n";
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			fp>>A[i*size + j];
		}
	}
	cout<<"Done!\n";
	fp.close();
	cout<<"Close the file!\n";
	cout<<"====================================\n\n";

	cout<<"Start to convert the dense matrix to sparse matrix with CSR!\n";

	start = clock();
	count = 0;
	for(int i = 0; i < size; i++)
	{
		rowptr[i] = count;
		count_nonz = 0;
		for(int j = 0; j < size; j++)
		{
			if(A[i*size + j] != 0)
			{
				value[count] = A[i*size + j];
				colidx[count] = j;
				count_nonz ++;
				count++;
			}
		}
	}
	rowptr[size] = count_nonz;
	end = clock();

	cout<<"Done in "<<(end - start) / CLOCKS_PER_SEC<<"secs.\n";

	cout<<"====================================\n\n";

	char file[] = "CSR_cpu.txt";

	fp.open(file, ios::out);
	if(!fp)
	{
		cout<<"Fail to open "<<file<<" !\n";
		return 0;
	}
	else
		cout<<"Start to write "<<file<<" !\n\n";

	for(int i = 0; i < count; i++)
	{
		if(i < count - 1)
		{
			fp<<value[i]<<",";
		}
		else
		{
			fp<<value[i]<<endl;
		}
	}

	for(int i = 0; i < count; i++)
	{
		if(i < count - 1)
		{
			fp<<colidx[i]<<",";
		}
		else
		{
			fp<<colidx[i]<<endl;
		}
	}

	for(int i = 0; i < size + 1; i++)
	{
		if(i < size)
		{
			fp<<rowptr[i]<<",";
		}
		else
		{
			fp<<count<<endl;
		}
	}

	fp.close();
	cout<<"Close the file!\n";

	return 0;
}