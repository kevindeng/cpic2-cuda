

int floatArrayCompare(float* a, float* b, int n, char* name1, char* name2, float epsilon);
void* copyFromGPU(void* g, int n);
void* copyToGPU(void* a, int n);
void copyFromGPU2(void* h, void* g, int n);
void copyToGPU2(void* g, void* h, int n);
void freeOnGPU(void* g);
int* createMutexes(int n);
void syncAddFloat(float* target, float amt, int* mutex);

void cgpost2l_cuda(float* part, float* q, float qm, int nop, int idimp,
            int nxv, int nyv, int npx, int npy, int* mutexes);

void caguard2l_cuda(float* q, int nx, int ny, int nxe, int nye);