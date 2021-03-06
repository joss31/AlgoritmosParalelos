#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

#define BlockSize 40

int** crear_matriz(int f,int c){
    int**mat;
    mat =new int *[f];
    for (int i=0;i<f;i++){
        mat[i]=new int [c];
    }
    return mat;
}

void llenar_rand (int** mat,int f,int c){
    for(int i=0;i<f;i++){
        for(int j=0;j<c;j++){
            mat[i][j]= rand()%100;
        }
    }
}
void print (int** mat,int f,int c){
    for(int i=0;i<f;i++){
        for(int j=0;j<c;j++){
            cout<<mat[i][j]<<"  ";
        }
        cout<<endl;
    }
}

int** multiplicar6B(int** mat1, int f1, int c1, int** mat2, int f2, int c2){
    int**res=crear_matriz(f1,c2);
    for( int i1 = 0; i1 < f1; i1 += BlockSize){
        for( int j1 = 0; j1 < f2; j1 += BlockSize){
            for(int k1 = 0 ;k1 < c1; k1 += BlockSize){
                for(int i = i1; i < min(i1 + BlockSize, f1); ++i){
                    for(int j = j1; j < min(j1 + BlockSize, f2); ++j){
                        for(int k = k1; k < min(k1 + BlockSize, c1); ++k){
                            res[i][j] = res[i][j] + mat1[i][k] * mat2[k][j];
                        }
                    }
                }
            }
        }
    }
    return res;
}
int main()
{
    int f=1000;
    int c=1000;
    int **mat=crear_matriz(f,c);
    int **mat1=crear_matriz(f,c);
    int **res;
    
    clock_t start_time;
    clock_t final_time;
    double total_time;
    
    srand(time(0));

    
    llenar_rand(mat,f,c);
    llenar_rand(mat1,f,c);

    //cout<<"Matriz 1"<<endl;
    //print(mat,f,c);
    //cout<<endl;

    //cout<<"Matriz 2"<<endl;
    //print(mat1,f,c);
    cout<<"Matriz de "<< f << "x" << c;
    cout<<endl;
    
    start_time = clock();
    res = multiplicar6B(mat,f,c,mat1,f,c);
    final_time=clock();
    
    //cout<<"Resultado de la Multiplicacion de 6 Bucles"<<endl;
    //print(res,f,c);
    
    total_time = ((double)(final_time - start_time)) / CLOCKS_PER_SEC;
    cout<<endl;
    cout<<"tiempo: "<<total_time<<endl;


    return 0;
}
