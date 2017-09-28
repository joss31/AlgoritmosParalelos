#include <stdio.h>
#include <pthread.h>

double intervalWidth, intervalMidPoint, area = 0.0;
int numberOfIntervals;

pthread_mutex_t area_mutex = PTHREAD_MUTEX_INITIALIZER;

int interval, iCount;
double myArea = 0.0, result;


void myPartOfCalc(int myID){
   
   int myInterval;
   double myIntervalMidPoint;
   myArea = 0.0;
   
   for (myInterval = myID + 1; myInterval <= numberOfIntervals; myInterval += numberOfIntervals) {
      myIntervalMidPoint = ((double) myInterval - 0.5) * intervalWidth;
      myArea += (4.0 / (1.0 + myIntervalMidPoint * myIntervalMidPoint));   
   } 
   
   result = myArea * intervalWidth;
   
   printf("\n %d: MiArea: %f.", myID, myArea);
   
   pthread_mutex_lock(&area_mutex);
   area += result;
   pthread_mutex_unlock(&area_mutex);
   
} 

main(int argc, char *argv[]){
   
      pthread_t * threads;
      pthread_mutex_t area_mutex = PTHREAD_MUTEX_INITIALIZER;
   
      if (argc < 2) {
         printf(" Usage: pi n.\n n is number of intervals.\n");
         exit(-1);
      } 
      
      numberOfIntervals = abs(atoi(argv[1]));
   
      if (numberOfIntervals == 0)
         numberOfIntervals = 50;
   
      printf("\n No. de Intervalos: %d", numberOfIntervals);
      
      threads = (pthread_t *) malloc(sizeof(pthread_t) * numberOfIntervals);
   
      intervalWidth = 1.0 / (double) numberOfIntervals;
   
      for (iCount = 0; iCount < numberOfIntervals; iCount++)
         pthread_create(&threads[iCount], NULL, (void *(*) (void *)) myPartOfCalc, (void *) iCount);
   
      for (iCount = 0; iCount < numberOfIntervals; iCount++)
         pthread_join(threads[iCount], NULL);
      
      printf("\n El valor Calculado de Pi%.15f\n", area);
   
      
      return 0;
   
} 
