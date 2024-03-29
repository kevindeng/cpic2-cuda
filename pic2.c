/*---------------------------------------------------------------------*/
/* Skeleton 2D Electrostatic PIC code */
/* written by Viktor K. Decyk, UCLA */
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <sys/time.h>
#include "push2.h"
#include "push2cuda.h"

#define TS {dtimer(&dtime,&itime,-1);}
#define TE(A) {dtimer(&dtime,&itime,1);time=(float)dtime;A+=time;}
#define TES(A) {TE(A);TS}
#define VALIDATE 0

void dtimer(double *time, struct timeval *itime, int icntrl);

int main(int argc, char *argv[]) 
{
    int indx = 6, indy = 7, npx = 48, npy = 12;
    float tend = 65.0f, dt = 0.1f;
    
    // parse in parameters
    if(argc == 7)
    {
        indx = atoi(argv[1]);
        indy = atoi(argv[2]);
        npx = atoi(argv[3]);
        npy = atoi(argv[4]);
        tend = atof(argv[5]);
        dt = atof(argv[6]);
        
        if(!(indx && indy && npx && npy && tend && dt))
        {
            printf("One or more parameters are invalid.\n");
            exit(1);
        }
    }
    else if(argc != 1)
    {
        printf("Usage: %s indx indy npx npy tend dt\n", argv[0]);
        exit(1);
    }
    else
    {
        printf("Using default parameters...\n");
    }
    
    int ndim = 2; 
    float qme = -1.0;
    float vtx = 1.0, vty = 1.0, vx0 = 0.0, vy0 = 0.0;
    float ax = .912871, ay = .912871;
    /* idimp = dimension of phase space = 4 */
    /* sortime = number of time steps between standard electron sorting */
    int idimp = 4, ipbc = 1, sortime = 50;
    float wke = 0.0, we = 0.0, wt = 0.0;
    /* declare scalars for standard code */
    int j;
    int np, nx, ny, nxh, nyh, nxe, nye, nxeh, nxyh, nxhy;
    int ny1, ntime, nloop, isign;
    float qbme, affp;
    /* declare arrays for standard code */
    float *part = NULL, *part2 = NULL, *tpart = NULL;
    float *qe = NULL;
    float *fxye = NULL;
    float complex *ffc = NULL;
    int *mixup = NULL;
    float complex *sct = NULL;
    int *npicy = NULL;
    /* declare and initialize timing data */
    float time;
    struct timeval itime;
    float tdpost = 0.0, tguard = 0.0, tfft = 0.0, tfield = 0.0;
    float tpush = 0.0, tsort = 0.0;
    double dtime;
    
    /* initialize scalars for standard code */
    np = npx*npy; nx = 1L<<indx; ny = 1L<<indy; nxh = nx/2; nyh = ny/2;
    nxe = nx + 2; nye = ny + 1; nxeh = nxe/2;
    nxyh = (nx > ny ? nx : ny)/2; nxhy = nxh > ny ? nxh : ny;
    ny1 = ny + 1;
    nloop = tend/dt + .0001; ntime = 0;
    qbme = qme;
    affp = (float) (nx*ny)/(float ) np;
    /* allocate and initialize data for standard code */
    part = (float *) malloc(idimp*np*sizeof(float));
    part2 = (float *) malloc(idimp*np*sizeof(float));
    qe = (float *) malloc(nxe*nye*sizeof(float));
    fxye = (float *) malloc(ndim*nxe*nye*sizeof(float));
    ffc = (float complex *) malloc(nxh*nyh*sizeof(float complex));
    mixup = (int *) malloc(nxhy*sizeof(int));
    sct = (float complex *) malloc(nxyh*sizeof(float complex));
    npicy = (int *) malloc(ny1*sizeof(int));
    /* prepare fft tables */
    cwfft2rinit(mixup,sct,indx,indy,nxhy,nxyh);
    /* calculate form factors */
    isign = 0;
    cpois22((float complex *)qe,(float complex *)fxye,isign,ffc,ax,ay,affp,
            &we,nx,ny,nxeh,nye,nxh,nyh);
    /* initialize electrons */
    cdistr2(part,vtx,vty,vx0,vy0,npx,npy,idimp,np,nx,ny,ipbc);
    
    
/* --------------------------------------------------------------------------*/
/* ---------------------------- set up ------------------------------------- */
/* --------------------------------------------------------------------------*/

    int sz_qe = nxe * nye * sizeof(float);
    int sz_part = idimp * np * sizeof(float);
    int sz_fxye = ndim*nxe*nye*sizeof(float);
    
    float* g_part = (float*)copyToGPU(part, sz_part);
    float* g_qe = (float*)copyToGPU(qe, sz_qe);
    float* g_fxye = (float*)copyToGPU(fxye, sz_fxye);
    float* g_wke = (float*)copyToGPU(&wke, sizeof(float));
    int* mutexes = createMutexes(nxe * nye);
    
/* --------------------------------------------------------------------------*/
    
    if(VALIDATE)
    {
        float* t = (float*)copyFromGPU(g_part, sz_part);
        float* t2 = (float*)copyFromGPU(g_qe, sz_qe);
        if(floatArrayCompare(t, part, sz_part / sizeof(float), "copy", "orig", 0) != 0 ||
           floatArrayCompare(t2, qe, sz_qe / sizeof(float), "copy", "orig", 0) !=0)
        {
            printf("Copying to and from GPU failed validation.\n");
            exit(1);
        }
        free(t);
    }
    
    
    
    
    /* * * * start main iteration loop * * * */
    
    L500: if (nloop <= ntime)
    goto L2000;
    /*    printf("ntime = %i\n",ntime); */
    
    
    
    
    /* deposit charge with standard procedure: updates qe */
    TS;
    cgpost2l_cuda(g_part, g_qe, qme, np, idimp, nxe, nye, npx, npy, mutexes);
    TE(tdpost);
    
    for (j = 0; j < nxe*nye; j++)
        qe[j] = 0.0;
    cgpost2l(part,qe,qme,np,idimp,nxe,nye);
    
    if(VALIDATE)
    {
        float* t = (float*)copyFromGPU(g_qe, sz_qe);
        if(floatArrayCompare(t, qe, sz_qe / sizeof(float), "gpu", "cpu", 1e-4) != 0)
        {
            printf("cgpost2l failed validation, ntime=%d\n", ntime);
            exit(1);
        }
        free(t);
    }
    
    
    
    /* add guard cells with standard procedure: updates qe */
    TS;
    caguard2l_cuda(g_qe,nx,ny,nxe,nye);
    TE(tguard);
    
    caguard2l(qe,nx,ny,nxe,nye);
    
    if(VALIDATE)
    {
        float* t = (float*)copyFromGPU(g_qe, sz_qe);
        if(floatArrayCompare(t, qe, sz_qe / sizeof(float), "gpu", "cpu", 1e-4) != 0)
        {
            printf("caguard2l failed validation, ntime=%d\n", ntime);
            exit(1);
        }
        free(t);
    }
    
    
    
    /* transform charge to fourier space with standard procedure: updates qe */
    TS;
    copyFromGPU2(qe, g_qe, sz_qe);
    isign = -1;
    cwfft2rx((float complex *)qe,isign,mixup,sct,indx,indy,nxeh,nye,
             nxhy,nxyh);
    copyToGPU2(g_qe, qe, sz_qe);
    TE(tfft);
    
    /* calculate force/charge in fourier space with standard procedure: */
    /* updates fxye                                                     */
    TS;
    isign = -1;
    cpois22((float complex *)qe,(float complex *)fxye,isign,ffc,ax,ay,
            affp,&we,nx,ny,nxeh,nye,nxh,nyh);
    TE(tfield);
    
    
    
    /* transform force to real space with standard procedure: updates fxye */
    TS;
    isign = 1;
    cwfft2r2((float complex *)fxye,isign,mixup,sct,indx,indy,nxeh,nye,
             nxhy,nxyh);
    TE(tfft);
    
    
    
    /* copy guard cells with standard procedure: updates fxye */
    TS;
    ccguard2l(fxye,nx,ny,nxe,nye);
    TE(tguard);
    
    
    
    /* push particles with standard precision: updates part, wke */
    TS;
    copyToGPU2(g_fxye, fxye, sz_fxye);
    cgpush2l_cuda(g_part,g_fxye,qbme,dt,g_wke,idimp,np,nx,ny,nxe,nye,ipbc,npx,npy,mutexes);
    TE(tpush);
    
    wke = 0.0;
    cgpush2l(part,fxye,qbme,dt,&wke,idimp,np,nx,ny,nxe,nye,ipbc);
    
    if(VALIDATE)
    {
        float* t = (float*)copyFromGPU(g_part, sz_part);
        if(floatArrayCompare(t, part, sz_part / sizeof(float), "gpu", "cpu", 1e-4) != 0)
        {
            printf("cgpush2l failed sdfsdf validation, ntime=%d\n", ntime);
            exit(1);
        }
        free(t);
    }
    
    /* sort particles by cell for standard code */
    if (sortime > 0) {
        if (ntime%sortime==0) {
            TS;
            cdsortp2yl(part,part2,npicy,idimp,np,ny1);
            /* exchange pointers */
            tpart = part;
            part = part2;
            part2 = tpart;
            copyToGPU2(g_part, part, sz_part);
            TE(tsort);
        }
    }
    
    
    
    if (ntime==0) {
        printf("Initial Field, Kinetic and Total Energies:\n");
        printf("%e %e %e\n",we,wke,wke+we);
    }
    ntime += 1;
    goto L500;
    L2000:
    
    /* * * * end main iteration loop * * * */
    
    printf("ntime = %i\n",ntime);
    printf("Final Field, Kinetic and Total Energies:\n");
    printf("%e %e %e\n",we,wke,wke+we);
    printf("\n");
    printf("deposit time = %f\n",tdpost);
    printf("guard time = %f\n",tguard);
    printf("solver time = %f\n",tfield);
    printf("fft time = %f\n",tfft);
    printf("push time = %f\n",tpush);
    printf("sort time = %f\n",tsort);
    tfield += tguard + tfft;
    printf("total solver time = %f\n",tfield);
    time = tdpost + tpush + tsort;
    printf("total particle time = %f\n",time);
    wt = time + tfield;
    printf("total time = %f\n",wt);
    printf("\n");
    wt = 1.0e+09/(((float) nloop)*((float) np));
    printf("Push Time (nsec) = %f\n",tpush*wt);
    printf("Deposit Time (nsec) = %f\n",tdpost*wt);
    printf("Sort Time (nsec) = %f\n",tsort*wt);
    printf("Total Particle Time (nsec) = %f\n",time*wt);
    
    freeOnGPU(g_part);
    freeOnGPU(g_qe);
    freeOnGPU(g_fxye);
    return 0;
}
