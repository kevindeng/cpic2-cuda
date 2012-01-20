int floatArrayCompare(float* a, float* b, int n, char* name1, char* name2, float epsilon);
void* copyFromGPU(void* g, int n);
void* copyToGPU(void* a, int n);
void copyFromGPU2(void* h, void* g, int n);
void copyToGPU2(void* g, void* h, int n);
void freeOnGPU(void* g);