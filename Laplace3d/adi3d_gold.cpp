
////////////////////////////////////////////////////////////////////////////////
//
// tridiagonal solver
//

void trid(float* a, float* b, float* c, float* d, int NX, int inc)
{
  int   i, ind=0;
  float aa, bb, cc, dd, c2[256], d2[256];

  //
  // forward pass
  //

  bb    = 1.0f/b[ind];
  cc    = bb*c[ind];
  dd    = bb*d[ind];
  c2[0] = cc;
  d2[0] = dd;

  for (i=1; i<NX; i++) {
    ind   = ind + inc;
    aa    = a[ind];
    bb    = b[ind] - aa*cc;
    dd    = d[ind] - aa*dd;
    bb    = 1.0f/bb;
    cc    = bb*c[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;
  }

  //
  // reverse pass
  //

  d[ind] = dd;

  for (i=NX-2; i>=0; i--) {
    ind = ind - inc;
    dd  = d2[i] - c2[i]*dd;
    d[ind] = dd;
  }
}


void Gold_adi(int NX, int NY, int NZ, float lam, float* u, float* du,
              float* ax, float* bx, float* cx,
              float* ay, float* by, float* cy,
              float* az, float* bz, float* cz ) 
{
  int   i, j, k, ind;
  float a, b, c, d;

  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
          d = 0.0f;          // Dirichlet b.c.'s
          a = 0.0f;
          b = 1.0f;
          c = 0.0f;
        }
        else {
          d = lam * ( u[ind-1    ] + u[ind+1    ]
                    + u[ind-NX   ] + u[ind+NX   ]
                    + u[ind-NX*NY] + u[ind+NX*NY] - 6.0f*u[ind]); 
          a = -0.5f*lam;
          b =  1.0f + lam;
          c = -0.5f*lam;
        }

        du[ind] = d;
        ax[ind] = a;
        bx[ind] = b;
        cx[ind] = c;
        ay[ind] = a;
        by[ind] = b;
        cy[ind] = c;
        az[ind] = a;
        bz[ind] = b;
        cz[ind] = c;
      }
    }
  }

  //
  // perform tri-diagonal solves in x-direction
  //

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      ind = j*NX + k*NX*NY;
      trid(&ax[ind],&bx[ind],&cx[ind],&du[ind],NX,1);
    }
  }

  //
  // perform tri-diagonal solves in y-direction
  //

  for (i=0; i<NX; i++) {
    for (k=0; k<NZ; k++) {
      ind = k*NX*NY + i;
      trid(&ay[ind],&by[ind],&cy[ind],&du[ind],NY,NX);
    }
  }

  //
  // perform tri-diagonal solves in z-direction, and update solution
  //

  for (j=0; j<NY; j++) {
    for (i=0; i<NX; i++) {
      ind = i + j*NX;
      trid(&az[ind],&bz[ind],&cz[ind],&du[ind],NZ,NX*NY);

      for (k=0; k<NZ; k++) {
        u[ind] += du[ind];
        ind    += NX*NY;
      }
    }
  }

}


