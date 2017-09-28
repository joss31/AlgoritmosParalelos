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
    int LIMIT=atoi(argv[1]);
    srand(time(NULL)+my_rank);
    int random = rand() % LIMIT;
    printf("number %d: %d\n", my_rank, random);  
    MPI_Barrier(MPI_COMM_WORLD);
    int sum=random;
    int potencia=1;
    int rec;
    while(potencia<comm_sz)
    {
        if(my_rank%(potencia<<1)==0)
        {
            if((my_rank+potencia<comm_sz))
            {
                MPI_Recv(&rec, 1, MPI_INT, my_rank+potencia,potencia, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum+=rec;
            }
        }
        else if(my_rank%(potencia<<1)==potencia)
        {
            rec=sum;
            MPI_Send(&rec, 1, MPI_INT, my_rank-potencia, potencia,MPI_COMM_WORLD);
        }
        else
            break;
        potencia=potencia<<1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank==0)
    {
        rec=sum;
        for(unsigned int i=1;i<comm_sz;++i)
            MPI_Send(&rec, 1, MPI_INT, i, 0,MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(&rec, 1, MPI_INT,0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum=rec;
    }
    printf("SUMA %d : %d\n",my_rank,sum);
    MPI_Finalize();
    return 0;
}
