#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 10000

int main()
{
	int sum = 0;
	double x, y;
	double start, end;

	start = clock();
	for (int i = 0; i < N; i++)
	{
		x = (double) rand() / RAND_MAX;
		y = (double) rand() / RAND_MAX;
		if(x*x + y*y < 1)
			sum++;
	}
	end = clock();

	printf("PI = %f\n", (double) 4 * sum / (N - 1));
	printf("Cost time %lf sec. \n", (double)(end - start)/CLOCKS_PER_SEC);

	return 0;
}