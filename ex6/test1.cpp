#include <mpi.h>
#include <stdio.h>
#define SIZE 10000000
#define PING 0
#define PONG 1

int main(int argc, char *argv[])
{
	int my_rank;
	int size;
	float buffer[SIZE];
	double start, end;
	MPI_Status status;

	MPI_Init(&argc, &argv);

	// Get the size of the rank
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of each process
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/*
	if(my_rank == 0)
	{
		printf("start program\n");
	}
	*/

	start = MPI_Wtime();
	if (my_rank == 0)
	{
		MPI_Send(buffer, SIZE, MPI_FLOAT, 1, PING, MPI_COMM_WORLD);
		MPI_Recv(buffer, SIZE, MPI_FLOAT, 1, PONG, MPI_COMM_WORLD, &status);
	}
	if (my_rank == 1)
	{
		MPI_Recv(buffer, SIZE, MPI_FLOAT, 0, PING, MPI_COMM_WORLD, &status);
		MPI_Send(buffer, SIZE, MPI_FLOAT, 0, PONG, MPI_COMM_WORLD);
	}
	end = MPI_Wtime();
	printf("Done in %lf secs.\n", end - start);

	printf("Rank %d says: Ping-pong is completed.\n", my_rank);

	printf("Hello from MPI rank %d in communicator with %d ranks. \n", my_rank, size);

	int input = my_rank*2 + 1, result;

	/* reduce the values of the different ranks in input to result of rank 0
	 * with the operation sum (max, logical and)
	*/
	MPI_Reduce(&input, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	//if (my_rank == 0)
		//printf("Rank 0 says: result is %i \n", result);

	MPI_Barrier(MPI_COMM_WORLD);

	// Compute sum of all ranks
	MPI_Allreduce(&input, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	printf("Rank %d says: result is %i \n", my_rank, result);

	MPI_Finalize();

	return 0;
}