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
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			if(A[i*n + j] != B[i*n + j]){
				return 0;
			}
		}
	}
	return 1;
	
}

int main(int argc, char **argv){
	
	clock_t t,ts,tc,tr;
	int rank, size ;
	MPI_Status status;
	MPI_Request requestA,requestB ;
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
	float *personal_comp_last = (float*)malloc (sizeof (float) * n * (partition+n%num_proc));
	
	for(int i=0;i<n * (partition+n%num_proc);i++)
		personal_comp_last[i]=0 ;	
	for(int i=0;i<n*partition;i++)
		personal_comp[i]=0 ;
	
	printf("n: %d, num_proc: %d, partition: %d",n,num_proc,partition);
	if (rank == 0) {
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
		t = clock();
		ts = clock() ;
		for(int i = size-1; i > 0; i--){
			MPI_Isend(A,n*32, MPI_FLOAT, i, tagA, MPI_COMM_WORLD,&requestA);
			MPI_Isend(B,32*n, MPI_FLOAT, i, tagB, MPI_COMM_WORLD,&requestB);
		}
		ts = clock()-ts ;
		tc=clock() ;
		for(int i=rank;i<rank+partition;i++)
			for(int j=0;j<n;j++)
				for(int k=0;k<32;k++)
					C[i*n+j] +=A[i*32+k]*B[k*n+j] ;
				
		MPI_Wait(&requestA,&status) ;		
		MPI_Wait(&requestB,&status) ;		
		tc=clock() - tc ;
		tr=clock() ;
		MPI_Recv(personal_comp_last,n*(partition+n%num_proc), MPI_FLOAT,size-1, tagR, MPI_COMM_WORLD, &status);
		for(int i=0;i<(partition+n%num_proc);i++)
			for(int j=0;j<n;j++)
				C[(i+(size-1)*partition)*n+j]=personal_comp_last[i*n+j] ;
			
		for(int r = size-2; r >0; r--){
			MPI_Recv(personal_comp,n*partition, MPI_FLOAT,r, tagR, MPI_COMM_WORLD, &status);
			for(int i=0;i<partition;i++)
				for(int j=0;j<n;j++)
					C[(i+r*partition)*n+j]=personal_comp[i*n+j] ;
		}		
		tr=clock() - tr ;
		t=clock()-t ;
		multiply_serial(A,B,C_serial,n,32,n);
		int is_eq = IsEqual(C,C_serial,n,n);
		printf("\n*******************************Equality: %d *****************************\n",is_eq);
		// printf("C Parallel: \n") ;
		// for(int i=0;i<n;i++){
			// for(int j=0;j<n;j++)
				// printf("%f ",C[i*n+j]) ;
			// printf("\n") ;
		// }
		// printf("C Serial: \n") ;
		// for(int i=0;i<n;i++){
			// for(int j=0;j<n;j++)
				// printf("%f ",C_serial[i*n+j]) ;
			// printf("\n") ;
		// }
		
	}
	else if(rank!=num_proc-1){
		MPI_Recv(A,n*32, MPI_FLOAT, 0, tagA, MPI_COMM_WORLD,&status);
		t=clock() ;
		MPI_Recv(B,32*n, MPI_FLOAT, 0, tagB, MPI_COMM_WORLD,&status);
		for(int i=0;i<partition;i++)
			for(int j=0;j<n;j++)
				for(int k=0;k<32;k++)
					personal_comp[i*n+j] +=A[(i+rank*partition)*32+k]*B[k*n+j] ;
		MPI_Send(personal_comp,n*partition, MPI_FLOAT,0, tagR, MPI_COMM_WORLD);
		t=clock()-t ;
	}
	else{
		MPI_Recv(A,n*32, MPI_FLOAT, 0, tagA, MPI_COMM_WORLD,&status);
		t=clock() ;
		MPI_Recv(B,32*n, MPI_FLOAT, 0, tagB, MPI_COMM_WORLD,&status);
		
		for(int i=0;i<partition+n%num_proc;i++)
			for(int j=0;j<n;j++)
				for(int k=0;k<32;k++)
					personal_comp_last[i*n+j] +=A[(i+rank*partition)*32+k]*B[k*n+j] ;
				
		MPI_Send(personal_comp_last,n*(partition+n%num_proc), MPI_FLOAT,0, tagR, MPI_COMM_WORLD);
		t=clock()-t ;
	}
	//printf("Message from process %d : %.13s\n", rank, message);
	MPI_Finalize();
	double time_taken = ((double)t)/CLOCKS_PER_SEC *1000;
	printf("\nTime taken in thread %d: %f milliseconds \n",rank,time_taken) ;
	if(rank==0)
		printf("*************Time taken in thread to send,calc,recv: %f,%f,%f\n",((double)ts)/CLOCKS_PER_SEC *1000,((double)tc)/CLOCKS_PER_SEC *1000,((double)tr)/CLOCKS_PER_SEC *1000) ;
	return 0;
} 