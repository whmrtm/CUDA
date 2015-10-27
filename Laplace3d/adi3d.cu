//
// Program to perform ADI time-marching on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

////////////////////////////////////////////////////////////////////////
// define kernel block size for 
////////////////////////////////////////////////////////////////////////

//#define BLOCK_X 32
//#define BLOCK_Y 4

#define BLOCK_X 32
#define BLOCK_Y 8

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <adi3d_kernel.cu>

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void Gold_adi(int, int, int, float, float*, float*, float*,  float*,
        float*, float*,  float*,  float*,  float*,  float*,  float*);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){

  // 'h_' prefix - CPU (host) memory space

  int    NX=256, NY=256, NZ=256, REPEAT=10, i, j, k, ind, printout=0;
  float  *h_u1, *h_u2, *h_du,
         *h_ax, *h_bx, *h_cx, 
         *h_ay, *h_by, *h_cy, 
         *h_az, *h_bz, *h_cz, 
         err, lam=1.0f;

  double timer, elapsed;

  // 'd_' prefix - GPU (device) memory space

  float  *d_u,  *d_du,
         *d_ax, *d_bx, *d_cx, 
         *d_ay, *d_by, *d_cy, 
         *d_az, *d_bz, *d_cz;


  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  if( NX>256 || NY>256 || NZ>256 ) {
    printf("No dimension can exceed 256 due to hard-coded array sizes\n");
    return -1;
  }

  // initialise card

  cutilDeviceInit(argc, argv);
 
  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_du = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_ax = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_bx = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_cx = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_ay = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_by = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_cy = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_az = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_bz = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_cz = (float *)malloc(sizeof(float)*NX*NY*NZ);

  cudaSafeCall(cudaMalloc((void **)&d_u,  sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_du, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_ax, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_bx, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_cx, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_ay, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_by, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_cy, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_az, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_bz, sizeof(float)*NX*NY*NZ) );
  cudaSafeCall(cudaMalloc((void **)&d_cz, sizeof(float)*NX*NY*NZ) );

  // initialise u1
    
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device

  elapsed_time(&timer);
  cudaSafeCall(cudaMemcpy(d_u, h_u1, sizeof(float)*NX*NY*NZ,
                           cudaMemcpyHostToDevice));
  cudaSafeCall(cudaDeviceSynchronize());
  elapsed = elapsed_time(&timer);
  printf("\nCopy u1 to device: %f (s) \n", elapsed);

  // Set up the execution configuration

  dim3 dimGrid1(1+(NX-1)/BLOCK_X, 1+(NY-1)/BLOCK_Y);
  dim3 dimBlock1(BLOCK_X,BLOCK_Y);

//  dim3 dimGrid2(1+(NX-1)/16, 1+(NY-1)/4);
//  dim3 dimBlock2(16,4);

  dim3 dimGrid2(1+(NX-1)/32, 1+(NY-1)/4);
  dim3 dimBlock2(32,4);

  dim3 dimGrid3(NY,NZ);
  dim3 dimBlock3(256);

  // Execute GPU kernel

  double time1=0, time2=0, time3=0, time4=0;

  for (i = 1; i <= REPEAT; ++i) {
    GPU_adi_rhs<<<dimGrid1, dimBlock1>>>(NX, NY, NZ,
                                         lam,  d_u,  d_du,
                                         d_ax, d_bx, d_cx,
                                         d_ay, d_by, d_cy,
                                         d_az, d_bz, d_cz);
    cudaCheckMsg("GPU_adi_rhs execution failed\n");

  cudaSafeCall(cudaDeviceSynchronize());
  time1 += elapsed_time(&timer);

  GPU_adi_x_float4_2<<<dimGrid1, dimBlock1>>>(NX, NY, NZ,
   (float4*)d_ax, (float4*)d_bx, (float4*)d_cx, (float4*)d_du);

//  GPU_adi_x<<<dimGrid2, dimBlock2>>>(NX, NY, NZ,
//                                     d_ax, d_bx, d_cx, d_du);
//    GPU_adi_x_new<<<dimGrid3, dimBlock3>>>(NX, NY, NZ,
//                                       d_ax, d_bx, d_cx, d_du);
    cudaCheckMsg("GPU_adi_x execution failed\n");

  cudaSafeCall(cudaDeviceSynchronize());
  time2 += elapsed_time(&timer);

    GPU_adi_y<<<dimGrid2, dimBlock2>>>(NX, NY, NZ,
                                       d_ay, d_by, d_cy, d_du);
    cudaCheckMsg("GPU_adi_y execution failed\n");

  cudaSafeCall(cudaDeviceSynchronize());
  time3 += elapsed_time(&timer);

    GPU_adi_z<<<dimGrid2, dimBlock2>>>(NX, NY, NZ, d_u,
                                       d_az, d_bz, d_cz, d_du);
    cudaCheckMsg("GPUadi_z execution failed\n");

  cudaSafeCall(cudaDeviceSynchronize());
  time4 += elapsed_time(&timer);

  }

  printf("\n%dx GPU_adi: %f (s) %f (s) %f (s) %f (s) \n",
         REPEAT, time1, time2, time3, time4);

  // Read back GPU results

  cudaSafeCall(cudaMemcpy(h_u2, d_u, sizeof(float)*NX*NY*NZ,
                           cudaMemcpyDeviceToHost) );
  elapsed = elapsed_time(&timer);
  printf("\nCopy u2 to host: %f (s) \n", elapsed);

  // print out corner of array

  if (printout) {
    for (k=0; k<3; k++) {
      for (j=0; j<8; j++) {
        for (i=0; i<8; i++) {
          ind = i + j*NX + k*NX*NY;
          printf(" %5.2f ", h_u2[ind]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }

  // Gold treatment

  for (int i = 1; i <= REPEAT; ++i) {
    Gold_adi(NX, NY, NZ, 
             lam,  h_u1, h_du,
             h_ax, h_bx, h_cx,
             h_ay, h_by, h_cy,
             h_az, h_bz, h_cz);
  }

  elapsed = elapsed_time(&timer);
  printf("\n%dx Gold_adi: %f (s) \n \n", REPEAT, elapsed);

  // print out corner of array

  if (printout) {
    for (k=0; k<3; k++) {
      for (j=0; j<8; j++) {
        for (i=0; i<8; i++) {
          ind = i + j*NX + k*NX*NY;
          printf(" %5.2f ", h_u1[ind]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }

  // error check

  err = 0.0;

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;
        err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
      }
    }
  }

  printf("\n rms error = %f \n",sqrt(err/ (float)(NX*NY*NZ)));

 // Release GPU and CPU memory

  cudaSafeCall(cudaFree(d_u) );
  cudaSafeCall(cudaFree(d_du));
  cudaSafeCall(cudaFree(d_ax));
  cudaSafeCall(cudaFree(d_bx));
  cudaSafeCall(cudaFree(d_cx));
  cudaSafeCall(cudaFree(d_ay));
  cudaSafeCall(cudaFree(d_by));
  cudaSafeCall(cudaFree(d_cy));
  cudaSafeCall(cudaFree(d_az));
  cudaSafeCall(cudaFree(d_bz));
  cudaSafeCall(cudaFree(d_cz));
  free(h_u1);
  free(h_u2);
  free(h_du);
  free(h_ax);
  free(h_bx);
  free(h_cx);
  free(h_ay);
  free(h_by);
  free(h_cy);
  free(h_az);
  free(h_bz);
  free(h_cz);

  cudaDeviceReset();
}
