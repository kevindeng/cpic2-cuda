extern "C"
{
    #include <stdlib.h>
    #include <stdio.h>
    #include <complex.h>
    #include <math.h>
    #include "push2cuda.h"
}

#define HANDLE_ERROR(A,M) {cudaError e;e=A;if(e!=cudaSuccess){printf("CUDA error: %s\n",M);exit(1);}}

int floatArrayCompare(float* a, float* b, int n, char* name1, char* name2, float epsilon)
{
    int failed = 0;
    for(int i = 0; i < n; i++)
    {
        float diff = fabs(a[i] - b[i]);
        if(diff > epsilon)
        {
            printf("%d: %s: %.7e %s: %.7e diff: %.7e\n", i, name1, name2, a[i], b[i], diff); 
            failed++;
        }
    }
    if(failed)
        printf("%d of %d failed\n", failed, n);
    return failed;
}

void* copyToGPU(void* a, int n)
{
    void* g;
    HANDLE_ERROR(cudaMalloc((void**)&g, n), "cudaMalloc in copyToGPU");
    HANDLE_ERROR(cudaMemcpy(g, a, n, cudaMemcpyHostToDevice), "cudaMemcpy in copyToGPU");
    return g;
}

void* copyFromGPU(void* g, int n)
{
    void* a = malloc(n);
    if(!a)
    {
        printf("malloc failed.");
        exit(1);
    }
    HANDLE_ERROR(cudaMemcpy(a, g, n, cudaMemcpyDeviceToHost), "cudaMemcpy in copyFromGPU");
    return a;
}

void copyToGPU2(void* g, void* h, int n)
{
    HANDLE_ERROR(cudaMemcpy(g, h, n, cudaMemcpyHostToDevice), "cudaMemcpy in copyToGPU2");
}

void copyFromGPU2(void* h, void* g, int n)
{
    HANDLE_ERROR(cudaMemcpy(h, g, n, cudaMemcpyDeviceToHost), "cudaMemcpy in copyFromGPU2");
}

void freeOnGPU(void* g)
{
    cudaFree(g);
}

int* createMutexes(int n)
{
    int* mut;
    HANDLE_ERROR(cudaMalloc((void**)&mut, n * sizeof(int)), "cudaMalloc in createMutexes");
    cudaMemset(mut, 0, n * sizeof(int));
    return mut;
}

__device__ void syncAddFloat(float* target, float amt, int* mutex)
{
    while(atomicCAS(mutex, 0, 1) != 0);
    (*target) += amt;
    atomicExch(mutex, 0);
}

/*---------------------------------------------------------------------------*/
/*--------------------------------- cgpost2l --------------------------------*/
/*---------------------------------------------------------------------------*/

__global__ void k_cgpost2l(float* part, float* q, float qm, int nop, int idimp,
            int nxv, int nyv, int* mutexes)
{
    int j, nn, mm, np, mp;
    float dxp, dyp, amx, amy;
    /* find interpolation weights */
    j = blockIdx.x + blockIdx.y * gridDim.x;
    if(j < nop) 
    //for(j = 0; j < nop; j++)
    {
        nn = part[idimp*j];
        mm = part[1+idimp*j];
        dxp = qm*(part[idimp*j] - (float) nn);
        dyp = part[1+idimp*j] - (float) mm;
        mm = nxv*mm;
        amx = qm - dxp;
        mp = mm + nxv;
        amy = 1.0 - dyp;
        np = nn + 1;
    /* deposit charge */
        syncAddFloat(q+np+mp, dxp*dyp, mutexes+np+mp);
        syncAddFloat(q+nn+mp, amx*dyp, mutexes+nn+mp);
        syncAddFloat(q+np+mm, dxp*amy, mutexes+np+mm);
        syncAddFloat(q+nn+mm, amx*amy, mutexes+nn+mm);
        /*q[np+mp] += dxp*dyp;
        q[nn+mp] += amx*dyp;
        q[np+mm] += dxp*amy;
        q[nn+mm] += amx*amy;*/
    }
}

void cgpost2l_cuda(float* part, float* q, float qm, int nop, int idimp,
            int nxv, int nyv, int npx, int npy, int* mutexes) 
{
    cudaMemset(q, 0, nxv * nyv * sizeof(float));
    dim3 grid(npx, npy);
    k_cgpost2l<<<grid, 1>>>(part, q, qm, nop, idimp, nxv, nyv, mutexes);
}



/*---------------------------------------------------------------------------*/
/*-------------------------------- caguard2l --------------------------------*/
/*---------------------------------------------------------------------------*/

__global__ void k_caguard2l(float* q, int nx, int ny, int nxe, int nye)
{
    int j, k;
    /* accumulate edges of extended field */
    j = blockIdx.x - ny;
    k = blockIdx.x;
    if(k < ny)
    {
        q[nxe*k] += q[nx+nxe*k];
        q[nx+nxe*k] = 0.0;
    }
    if(j >= 0 && j < nx)
    {
        q[j] += q[j+nxe*ny];
        q[j+nxe*ny] = 0.0;
    }
    if(k == 0)
    {
        q[0] += q[nx+nxe*ny];
        q[nx+nxe*ny] = 0.0;
    }
}

void caguard2l_cuda(float* q, int nx, int ny, int nxe, int nye) 
{
    k_caguard2l<<<nx + ny, 1>>>(q, nx, ny, nxe, nye);
}