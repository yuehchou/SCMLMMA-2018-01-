#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <omp.h>
using namespace std;

int main()
{
	int pb, n, temp = 1, P = 4;
	char ans;
	cout<<"1. calculate 2-norm of a vector>.\n";
	cout<<"2. calculate the matrix vector multiplication.\n";
	cout<<"3. the matrix multiplication.\n";
	cout<<"What are you want to do?(enter the number)\n";
	cin>>pb;
	cout<<"Do you wnat to use openMP?\n";
	cin>>ans;

	if(ans == 'n' || ans == 'N')
	{
		if (pb == 1)
		{
			cout<<"What times do you wnat to do?\n";
			cin>>n;

			vector<double> t(n);

			for (int i = 0; i < n; i++)
			{
				temp = temp*2;
				vector<double> x(temp);
				double norm_x = 0.0;

				double start = clock();
				cout<<"\nComputing ...\n";

				for(int j = 0; j < temp; j++)
					norm_x += x[j]*x[j];
				norm_x = sqrt(norm_x);

				double end = clock();

				t[i] = (end - start)/CLOCKS_PER_SEC;
				cout<<"Done at "<<i+1<<" times"<<endl;
			}

			fstream fp;

			if(!fp)
				cout<<"Fail to open file!\n";
			else
				cout<<"Success to open file!\n";

			fp.open( "ex1-1_time", ios::out);

			cout<<"Writting...\n";
			for(int i = 0; i < n; i++)
				fp<<t[i]<<endl;
			cout<<"Done\n";

			fp.close();
		}

		else if (pb == 2)
	    {
			cout<<"What times do you wnat to do?\n";
			cin>>n;

			vector<double> t(n);

			for(int i = 0; i < n; i++)
			{
				temp = temp*2;
	        	vector<double> x(temp), A(temp*temp), y(temp,0);

	        	double start = clock();
	        	cout<<"\nComputing ...\n";

	        	for(int j = 0; j < temp; j++)
					for(int k = 0; k < temp; k++)
						y[j] += A[j*n + k]*x[k];

				double end = clock();

				t[i] = (end - start)/CLOCKS_PER_SEC;
				cout<<"Done at "<<i+1<<" times"<<endl;
			}

			fstream fp;

			if(!fp)
				cout<<"Fail to open file!\n";
			else
				cout<<"Success to open file!\n";

			fp.open( "ex1-2_time", ios::out);

			cout<<"Writting...\n";
			for(int i = 0; i < n; i++)
				fp<<t[i]<<endl;
			cout<<"Done\n";

			fp.close();
		}

		else if (pb == 3)
	    {

			cout<<"What times do you wnat to do?\n";
			cin>>n;

			vector<double> t(n);

			for(int i = 0; i < n; i++)
			{
				temp = temp*2;
	            vector<double> A(temp*temp), B(temp*temp), C(temp*temp,0);

				double start = clock();                
				cout<<"\nComputing ...\n";
		                
				for(int j = 0; j < temp; j++)
					for(int k = 0; k < temp; k++)
						for(int l = 0; l < temp; l++)
							C[j*n + k] += A[j*n + l]*B[l*n + k];

				double end = clock();
				
				t[i] = (end - start)/CLOCKS_PER_SEC;
				cout<<"Done at "<<i+1<<" times"<<endl;
			}

			fstream fp;

			if(!fp)
				cout<<"Fail to open file!\n";
			else
				cout<<"Success to open file!\n";

			fp.open( "ex1-3_time", ios::out);

			cout<<"Writting...\n";
			for(int i = 0; i < n; i++)
				fp<<t[i]<<endl;
			cout<<"Done\n";

			fp.close();
		}

		else
		{
			cout<<"Your input number is illegal!\n";
		}
	}

	if(ans == 'y' || ans == 'Y')
	{
			if (pb == 1)
			{
				cout<<"What times do you wnat to do?\n";
				cin>>n;

				vector<double> t(n);

				for (int i = 0; i < n; i++)
				{
					temp = temp*2;
					vector<double> x(temp);
					double norm_x = 0.0;

					double start = clock();
					cout<<"\nComputing ...\n";
					#pragma omp parallel num_threads(P)
					{
						#pragma omp for
						for(int j = 0; j < temp; j++)
							norm_x += x[j]*x[j];
					}
					norm_x = sqrt(norm_x);

					double end = clock();

					t[i] = (end - start)/CLOCKS_PER_SEC;
					cout<<"Done at "<<i+1<<" times"<<endl;
				}

				fstream fp;

				if(!fp)
					cout<<"Fail to open file!\n";
				else
					cout<<"Success to open file!\n";

				fp.open( "ex1-1_omptime", ios::out);

				cout<<"Writting...\n";
				for(int i = 0; i < n; i++)
					fp<<t[i]<<endl;
				cout<<"Done\n";

				fp.close();
			}

			else if (pb == 2)
		    {
				cout<<"What times do you wnat to do?\n";
				cin>>n;

				vector<double> t(n);

				for(int i = 0; i < n; i++)
				{
					temp = temp*2;
		        	vector<double> x(temp), A(temp*temp), y(temp,0);

		        	double start = clock();
		        	cout<<"\nComputing ...\n";

		        	#pragma omp parallel num_threads(P)
					{
			        	#pragma omp for
			        	for(int j = 0; j < temp; j++)
							for(int k = 0; k < temp; k++)
								y[j] += A[j*n + k]*x[k];
					}

					double end = clock();

					t[i] = (end - start)/CLOCKS_PER_SEC;
					cout<<"Done at "<<i+1<<" times"<<endl;
				}

				fstream fp;

				if(!fp)
					cout<<"Fail to open file!\n";
				else
					cout<<"Success to open file!\n";

				fp.open( "ex1-2_omptime", ios::out);

				cout<<"Writting...\n";
				for(int i = 0; i < n; i++)
					fp<<t[i]<<endl;
				cout<<"Done\n";

				fp.close();
			}

			else if (pb == 3)
		    {

				cout<<"What times do you wnat to do?\n";
				cin>>n;

				vector<double> t(n);

				for(int i = 0; i < n; i++)
				{
					temp = temp*2;
		            vector<double> A(temp*temp), B(temp*temp), C(temp*temp,0);

					double start = clock();                
					cout<<"\nComputing ...\n";
			        
			        #pragma omp parallel num_threads(P)
					{ 
				        #pragma omp for       
						for(int j = 0; j < temp; j++)
							for(int k = 0; k < temp; k++)
								for(int l = 0; l < temp; l++)
									C[j*n + k] += A[j*n + l]*B[l*n + k];
					}

					double end = clock();
					
					t[i] = (end - start)/CLOCKS_PER_SEC;
					cout<<"Done at "<<i+1<<" times"<<endl;
				}

				fstream fp;

				if(!fp)
					cout<<"Fail to open file!\n";
				else
					cout<<"Success to open file!\n";

				fp.open( "ex1-3_omptime", ios::out);

				cout<<"Writting...\n";
				for(int i = 0; i < n; i++)
					fp<<t[i]<<endl;
				cout<<"Done\n";

				fp.close();
			}

			else
			{
				cout<<"Your input number is illegal!\n";
			}
	}

	

	return 0;
}
