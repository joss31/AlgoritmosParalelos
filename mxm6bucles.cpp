#include <iostream>
#include <random>
#include <time.h>
#include <math.h>

using namespace std;
#define n 16
#define BlockSize 4

int main() {
    
    double A[n][n], B[n][n], C[n][n];
    C[0][0]=0;
    srand(time(NULL));
    
    for( int i = 0; i < n; i++){
        for ( int j = 0; j < n; j++){
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }
    
    for( int i1 = 0; i1 < n; i1 += BlockSize)
    {
        for( int j1 = 0; j1 < n; j1 += BlockSize)
        {
            for(int k1 = 0 ;k1 < n; k1 += BlockSize)
            {
                for(int i = i1; i < min(i1 + BlockSize, n); ++i)
                {
                    for(int j = j1; j < min(j1 + BlockSize, n); ++j)
                    {
                        for(int k = k1; k < min(k1 + BlockSize, n); ++k)
                        {
                            C[i][j] = C[i][j] + A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
    
    for( int i = 0; i < n; i++){
        for ( int j = 0; j < n; j++){
            cout<< A[i][j] <<"|";
        }
        cout<<"\n------\n";
    }
    cout<<endl;
    
    for( int i = 0; i < n; i++){
        for ( int j = 0; j < n; j++){
            cout<< B[i][j] <<"|";
        }
        cout<<"\n------\n";
    }
    cout<<endl;
    
    for( int i = 0; i < n; i++){
        for ( int j = 0; j < n; j++){
            cout<< C[i][j] <<"|";
        }
        cout<<"\n------\n";
    }
    cout<<endl;
    return 0;
}