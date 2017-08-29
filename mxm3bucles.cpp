#include <iostream>
#include <random>
#include <time.h>

using namespace std;
#define MAX 16

int main() {
    
    double A[MAX][MAX], B[MAX][MAX], C[MAX][MAX];
    double tmp = 0;
    
    srand(time(NULL));
    
    for( int i = 0; i < MAX; i++){
        for ( int j = 0; j < MAX; j++){
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }
    
    for( int k = 0; k < MAX; ++k){
        for(int i = 0 ; i < MAX; ++i){
            for(int j = 0 ; j < MAX; ++j){
                tmp += A[k][j] * B[j][i];
            }
            C[k][i] = tmp;
            tmp = 0;
        }
    }
    
    for( int i = 0; i < MAX; i++){
        for ( int j = 0; j < MAX; j++){
            cout<< A[i][j] <<"|";
        }
        cout<<"\n------\n";
    }
    cout<<endl;
    
    for( int i = 0; i < MAX; i++){
        for ( int j = 0; j < MAX; j++){
            cout<< B[i][j] <<"|";
        }
        cout<<"\n------\n";
    }
    cout<<endl;
    
    for( int i = 0; i < MAX; i++){
        for ( int j = 0; j < MAX; j++){
            cout<< C[i][j] <<"|";
        }
        cout<<"\n------\n";
    }
    cout<<endl;
    return 0;
}