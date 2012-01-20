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
            failed = 1;
        }
    }
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
    cudaMemcpy(g, h, n, cudaMemcpyHostToDevice);
}

void copyFromGPU2(void* h, void* g, int n)
{
    cudaMemcpy(h, g, n, cudaMemcpyDeviceToHost);
}

void freeOnGPU(void* g)
{
    cudaFree(g);
}