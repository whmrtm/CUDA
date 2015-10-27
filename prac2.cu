
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cutil_inline.h>

#include <cuda.h>
#include <curand.h>


////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;


////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////


__global__ void pathcalc(float *d_z, float *d_v)
{
  float s1, s2, y1, y2, payoff;

  // move array pointers to correct position


  d_z = d_z + threadIdx.x + 2*N*blockIdx.x*blockDim.x;

  // version 2
  // d_z = d_z + 2*N*threadIdx.x + 2*N*blockIdx.x*blockDim.x;

  d_v = d_v + threadIdx.x +     blockIdx.x*blockDim.x;

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n=0; n<N; n++) {
    y1   = (*d_z);
    // version 1
    d_z += blockDim.x;      // shift pointer to next element
    // version 2
    // d_z += 1; 

    y2   = rho*y1 + alpha*(*d_z);
    // version 1
    d_z += blockDim.x;      // shift pointer to next element
    // version 2
    // d_z += 1; 

    s1 = s1*(con1 + con2*y1);
    s2 = s2*(con1 + con2*y2);
  }

  // put payoff value into device array

  payoff = 0.0f;
  if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f ) payoff = exp(-r*T);

  *d_v = payoff;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
    
  int     NPATH=960000, h_N=100;
  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float  *h_v, *d_v, *d_z;
  double  sum1, sum2;

  double timer, elapsed;  // timer variable and elapsed time

  curandGenerator_t gen;

  // initialise card

  cutilDeviceInit(argc, argv);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float)*NPATH);

  cudaSafeCall( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  cudaSafeCall( cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*NPATH) );

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.05f;
  h_sigma = 0.1f;
  h_rho   = 0.5f;
  h_alpha = sqrt(1.0f-h_rho*h_rho);
  h_dt    = 1.0f/h_N;
  h_con1  = 1.0f + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;

  cudaSafeCall( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  cudaSafeCall( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  cudaSafeCall( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  cudaSafeCall( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  cudaSafeCall( cudaMemcpyToSymbol(rho,  &h_rho,  sizeof(h_rho)) );
  cudaSafeCall( cudaMemcpyToSymbol(alpha,&h_alpha,sizeof(h_alpha)) );
  cudaSafeCall( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  cudaSafeCall( cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)) );
  cudaSafeCall( cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)) );

  // random number generation

  elapsed_time(&timer);  // initialise timer

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateNormal(gen, d_z, 2*h_N*NPATH, 0.0f, 1.0f);
 
  cudaSafeCall( cudaDeviceSynchronize() );

  elapsed = elapsed_time(&timer);
  printf("\n CURAND normal RNG execution time (s): %f ,   samples/sec: %e \n",
         elapsed, 2.0*h_N*NPATH/elapsed);

  // execute kernel and time it

  pathcalc<<<NPATH/64, 64>>>(d_z, d_v);
  cudaCheckMsg("pathcalc execution failed\n");
  cudaSafeCall( cudaDeviceSynchronize() );

  elapsed = elapsed_time(&timer);
  printf("Monte Carlo kernel execution time (s): %f \n",elapsed);

  // copy back results

  cudaSafeCall( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                 cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );



  curandDestroyGenerator(gen);

 
  free(h_v);
  cudaSafeCall( cudaFree(d_v) );
  cudaSafeCall( cudaFree(d_z) );


  cudaDeviceReset();

}
