#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Fonction auxiliaire de compute_string
void __global__ compute_string_kernel( char * res, char * a, char * b, char *c, int length ) 
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

for ( i = 0 ; i < length ; i++ ) 
{
res[i] = a[i] + b[i] + c[i] ; 
}
}

/* Function computing the final string to print */
void compute_string( char * res, char * a, char * b, char *c, int length ) 
{

// En reprenant l'exple de cours
// 1. Device pointers
char * d_a, * d_b, * d_c, * d_res ;

// 2. Memory allocation on GPU
cudaMalloc( (void**)&d_a, length * sizeof(char) ) ;
cudaMalloc( (void**)&d_b, length * sizeof(char) ) ;
cudaMalloc( (void**)&d_c, length * sizeof(char) ) ;
cudaMalloc( (void**)&d_res, length * sizeof(char) ) ;

// 3. Transfer to GPU
cudaMemcpy( d_a, a, length * sizeof(char), cudaMemcpyHostToDevice ) ;
cudaMemcpy( d_b, b, length * sizeof(char), cudaMemcpyHostToDevice ) ;
cudaMemcpy( d_c, c, length * sizeof(char), cudaMemcpyHostToDevice ) ;
cudaMemcpy( d_res, res, length * sizeof(char), cudaMemcpyHostToDevice ) ;

// 4. Kernel execution
compute_string_kernel<<<1, length>>>( d_res, d_a, d_b, d_c, length ) ;

// 5. Transfer to host
cudaMemcpy( res, d_res, length * sizeof(char), cudaMemcpyDeviceToHost ) ;

// 6. Free GPU memory
cudaFree( d_a ) ;
cudaFree( d_b ) ;
cudaFree( d_c ) ;
cudaFree( d_res ) ;
}

int main()
{

char * res ;

char a[30] = { 40, 70, 70, 70, 80, 0, 50, 80, 80, 70, 70, 0, 40, 80, 79, 
70, 0, 40, 50, 50, 0, 70, 80, 0, 30, 50, 30, 30, 0, 0 } ;
char b[30] = { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0 } ;
char c[30] = { 22, 21, 28, 28, 21, 22, 27, 21, 24, 28, 20, 22, 20, 24, 22, 
29, 22, 21, 20, 25, 22, 25, 20, 22, 27, 25, 28, 25, 0, 0 } ;

res = (char *)malloc( 30 * sizeof( char ) ) ;


/* This function call should be programmed in CUDA */
/* -> need to allocate and transfer data to/from the device */
compute_string( res, a, b, c, 30 ) ;

printf( "%s\n", res ) ;

free( res ) ;

return 0 ;
}
