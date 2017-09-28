//PI

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>


int main(int argc, char **argv)
{
	const long long int RAND_LIMIT=2000000;
	int REC=RAND_LIMIT/2;
	int comm_sz, my_rank;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if(argc!=2)
	{
		if(my_rank == 0)
			printf("error\n");
		MPI_Finalize();
		return 0;
	}
	int TOTAL_TOSSES=atoi(argv[1]);
	if(TOTAL_TOSSES%comm_sz!=0 )
	{
		if(my_rank == 0)
			printf("error\n");
		MPI_Finalize();
		return 0;
	}
	int rec=0;
	int size = TOTAL_TOSSES/(comm_sz);
	int local_sum=0;
	int number_in_circle=0;
	srand(time(NULL));
	for(unsigned int i=0;i<size;++i) // halla los valores aleatorios para estimar los valores de pi
	{
		double X=((double)(rand()%RAND_LIMIT))/REC-1;
		double Y=((double)(rand()%RAND_LIMIT))/REC-1;
		double distance_square=(X*X)+(Y*Y);
		if(distance_square <=1)
			local_sum++;
	}
	MPI_Reduce(&local_sum,&number_in_circle,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD); //Reduce un valor de un grupo de procesos en un único proceso raíz.
	if (my_rank == 0)
	{
		float pi_estimate = (4.0*number_in_circle)/(TOTAL_TOSSES);
		printf("%f\n", pi_estimate);
	}
	MPI_Finalize();
	return 0;
}
