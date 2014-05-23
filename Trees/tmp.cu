#include <Nutty.h>

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cudadevrt.lib")

__global__ void test(uint* ptr)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    ptr[idx] = idx;

    if(ptr)
    {
     //   test<<<1,1>>>(0);
    }
}


extern "C" void runKernel(void)
{
    test<<<1,1>>>(0);
}