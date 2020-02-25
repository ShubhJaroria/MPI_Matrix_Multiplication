#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>

void multiply_serial(float *A, float *B, float *C, int m, int n, int p){ // A is m x n and B is n x p
	int i, j, k;
	for (i = 0; i < m; i++){
		for (j = 0; j < p; j++){
			C[i*p + j] = 0;
			for (k = 0; k < n; k++)
				C[i*p + j] += A[i*n + k] * B[k*p + j];
		}
	}
}

int IsEqual(float *A, float *B, int m, int n){
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			if(A[i*n + j] != B[i*n + j])
				return 0;
	return 1;
}

int main(int argc, char **argv){
	
	clock_t t,ts,tc,tr;
	int rank, size ;
	
	MPI_Init(&argc,&argv) ;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int i, tagA = 69, tagB = 6969, tagR=696969;
	int num_proc = size;
	int n=atoi(argv[1]);
	
	float *A = (float*)malloc (sizeof (float) * n * 32); /* n is the length of the array */
	float *B = (float*)malloc (sizeof (float) * 32 * n);
	float *C_serial = (float*)malloc (sizeof (float) * n * n);
	float *C = (float*)malloc (sizeof (float) * n * n);
	
	int partition = (n/num_proc) ;
	float *personal_comp = (float*)malloc (sizeof (float) * n * partition);
	
	//zeros	
	for(int i=0;i<n*partition;i++)
		personal_comp[i]=0 ;
	
	//randomize initialisation
	printf("n: %d, num_proc: %d, partition: %d",n,num_proc,partition);
	if (rank == 0){
		float random;
		for(i = 0; i < 32*n; i++){
			random = (float) rand();
			random = random / RAND_MAX;
			A[i] = random;
		}
		for(i = 0; i < 32*n; i++){
			random = (float) rand();
			random = random / RAND_MAX;
			B[i] = random;
		}
		for(i = 0; i < n*n; i++){
			C[i] = 0.0;
			C_serial[i] = 0.0;
		}
	}
	
//Data transfer to all threads
	MPI_Bcast(A,32*n,MPI_FLOAT,0,MPI_COMM_WORLD) ;
	MPI_Bcast(B,32*n,MPI_FLOAT,0,MPI_COMM_WORLD) ;
	
	MPI_Barrier(MPI_COMM_WORLD) ;

//computation
	for(int i=0;i<partition;i++)
		for(int j=0;j<n;j++)
			for(int k=0;k<32;k++)
				personal_comp[i*n+j] +=A[(i+rank*partition)*32+k]*B[k*n+j] ;
	
	MPI_Barrier(MPI_COMM_WORLD) ;
	
//gather
	MPI_Gather(personal_comp,partition*n,MPI_FLOAT,C,partition*n,MPI_FLOAT,0,MPI_COMM_WORLD) ;
	
	MPI_Barrier(MPI_COMM_WORLD) ;

	MPI_Finalize();

	if(rank==0){
		multiply_serial(A,B,C_serial,n,32,n);
		int is_eq = IsEqual(C,C_serial,n,n);
		printf("\n*******************************Equality: %d *****************************\n",is_eq);		
	}
	return 0;
} 