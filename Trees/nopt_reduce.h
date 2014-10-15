#pragma once

template <
    uint blockSize, 
    typename T,
    typename Operator
>
__global__ void optblockReduce(const T* __restrict data, T* dstData, Operator op, T neutral, uint elemsPerBlock, uint N)
{
    uint tid = threadIdx.x;
    uint globalOffset = blockIdx.x * elemsPerBlock;
    uint i = threadIdx.x;

    __shared__ IndexedSAHSplit neutralSplit;
    neutralSplit.index = 0;
    neutralSplit.sah = FLT_MAX;

    IndexedSAHSplit split;
    split.index = 0;
    split.sah = FLT_MAX;

    while(i < elemsPerBlock && globalOffset + i < N) 
    { 
        split = ris(split, ris(data[globalOffset + i], globalOffset + i + blockSize < N ? data[globalOffset + i + blockSize] : neutral));
        i += 2 * blockSize;
    }

    split = blockReduceSum<blockSize/32>(split, neutralSplit);
    if(tid == 0) dstData[blockIdx.x] = split;

}

template <
    typename IteratorDst,
    typename IteratorSrc,
    typename BinaryOperation,
    typename T
>
__host__ void ReduceOpt(
IteratorDst& dst, 
IteratorSrc& src,
size_t d,
BinaryOperation op,
T neutral,
cudaStream_t pStream = NULL)
{
    const uint elementsPerBlock = 2048;//d / blockCount; //4 * 512;
    const uint blockSize = 256;//elementsPerBlock / 2;

    //assert(d > 1);
    uint grid = nutty::cuda::GetCudaGrid((uint)d, elementsPerBlock);
    if(grid > 1)//d > elementsPerBlock)
    {
        optblockReduce<blockSize><<<grid, blockSize, 0, pStream>>>(src(), dst(), op, neutral, elementsPerBlock, (uint)d);
        optblockReduce<blockSize><<<1, blockSize, 0, pStream>>>(dst(), dst(), op, neutral, grid, grid);
    }
    else
    {
        optblockReduce<blockSize><<<1, blockSize, 0, pStream>>>(src(), dst(), op, neutral, d, (uint)d);
    }
}


template <
    typename IteratorDst,
    typename IteratorSrc,
    typename BinaryOperation,
    typename T
>
__host__ void ReduceNoOpt(
IteratorDst& dst, 
IteratorSrc& src,
size_t d,
BinaryOperation op,
T neutral,
cudaStream_t pStream = NULL)
{

    nutty::base::Reduce1(dst, src, d, op, neutral, pStream);
    // #ifndef USE_OPT_RED
    //             const uint elementsPerBlock = 2*4096;//d / blockCount; //4 * 512;
    // #else
    //             const uint elementsPerBlock = 256;
    // #endif
    //     const uint blockSize = 256;//elementsPerBlock / 2;
    //     const uint elementsPerBlock = 2*4096;
    //     //assert(d > 1);
    //     uint grid = nutty::cuda::GetCudaGrid((uint)d, elementsPerBlock);
    //     if(grid > 1)//d > elementsPerBlock)
    //     {
    //         nutty::cuda::blockReduce<blockSize><<<grid, blockSize, 0, pStream>>>(src(), dst(), op, neutral, elementsPerBlock, (uint)d);
    //         nutty::cuda::blockReduce<blockSize><<<1, blockSize, 0, pStream>>>(dst(), dst(), op, neutral, grid, grid);
    //     }
    //     else
    //     {
    //         nutty::cuda::blockReduce<blockSize><<<1, blockSize, 0, pStream>>>(src(), dst(), op, neutral, d, (uint)d);
    //     }
}