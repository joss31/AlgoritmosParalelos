#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    int comm_sz, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(argc!=2)
    {
        if(my_rank==0)
            printf("error\n");
        MPI_Finalize();
        return 0;
    }
    int potencia=1;
    while(potencia<=comm_sz)
    {
        if(comm_sz%potencia)
        {
            if(my_rank==0)
                printf("error\n");
            MPI_Finalize();
            return 0;
        }
        potencia=potencia<<1;
    }
    potencia=1;
    int LIMIT=atoi(argv[1]);
    srand(time(NULL)+my_rank);
    int random = rand() % LIMIT;
    printf("number %d: %d\n", my_rank, random);
    MPI_Barrier(MPI_COMM_WORLD);
    int sum=random;
    int rec;
    while(potencia<comm_sz)
    {
        rec=sum;
        int hermano=((my_rank+potencia)%(potencia<<1))+(potencia<<1)*(my_rank/(potencia<<1));
        MPI_Send(&rec, 1, MPI_INT,hermano,potencia,MPI_COMM_WORLD);
        MPI_Recv(&rec, 1, MPI_INT,hermano,potencia,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        sum+=rec;
        potencia=potencia<<1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("SUMA %d : %d\n",my_rank,sum);
    MPI_Finalize();
    return 0;
}
