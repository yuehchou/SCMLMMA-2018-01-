#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define N 100000000

int main()
{
	struct timespec start, end;
	int sum = 0;
	double x, y;
	//double start, end;

	omp_set_num_threads(24);


	clock_gettime(CLOCK_REALTIME, &start);

	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < N; i++)
		{
			x = (double) rand() / RAND_MAX;
			y = (double) rand() / RAND_MAX;
			if(x*x + y*y < 1)
				sum++;
		}
	}
	clock_gettime(CLOCK_REALTIME, &end);

	printf("PI = %f\n", (double) 4 * sum / (N - 1));
	printf("Cost time %lf sec. \n", (end.tv_sec - start.tv_sec + (double)(end.tv_nsec - start.tv_nsec)/1e9));

	return 0;
}