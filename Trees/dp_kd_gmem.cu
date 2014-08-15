#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#undef NUTTY_DEBUG

#include "cuKDTree.h"

#if 0

#define EVENT_GROUP_SIZE 256U

#undef MEM_CHECK

#ifdef MEM_CHECK
#ifndef NUTTY_DEBUG
#define NUTTY_DEBUG
#endif
#endif

#include "kd_kernel.h"
#include "kd_scan_kernel.h"
#include "shared_kernel.h"
#include "shared_types.h"
#include <Reduce.h>

#include <Sort.h>
#include <Scan.h>
#include <queue>
#include <ForEach.h>
#include <Fill.h>
#include <cuda/Globals.cuh>
#include "buffer_print.h"
#include <chimera/Timer.h>
#include <fstream>
#include "device_buffer.cuh"

__device__ CTuint d_deviceError;
__device__ CTuint d_mem_alloc_error;

__device__ CTuint d_leafContentOffset;
__device__ CTuint d_currentEventCount;
__device__ CTuint d_maxLeafCountNextLevel;
__device__ CTuint d_currentLeafCount;
__device__ CTuint d_interiorNodesCountThisLevel;

__device__ CTbyte d_toggleIndex;

__device__ CTuint d_needMoreMemory[5];
#define needMoreEventMemory d_needMoreMemory[0]
#define needMoreNodeMemory d_needMoreMemory[1]
#define needMorePerLevelNodeMemory d_needMoreMemory[2]
#define needMoreLeafNodeMemory d_needMoreMemory[3]
#define needMoreLeafContentMemory d_needMoreMemory[4]


#define h_needMoreEventMemory h_needMoreMemory[0]
#define h_needMoreNodeMemory h_needMoreMemory[1]
#define h_needMorePerLevelNodeMemory h_needMoreMemory[2]
#define h_needMoreLeafNodeMemory h_needMoreMemory[3]
#define h_needMoreLeafContentMemory h_needMoreMemory[4]

__device__ CTreal msgBuffer[256];

template<typename T>
__device__ void __forceinline swap(T& a, T& b)
{
    T tmp = a; a = b; b = tmp;
}

struct MemorySizes
{
    CTuint nodeMemory;
    CTuint eventMemory;
    CTuint perLevelNodeMemory;
    CTuint leafContentMemory;
    CTuint leafNodeMemory;

    __host__ void print(void)
    {
        __ct_printf("\nMemorySizes: nodeMemory=%d eventMemory=%d perLevelNodeMemory=%d leafContentMemory=%d leafNodeMemory=%d\n",
            nodeMemory, eventMemory, perLevelNodeMemory, leafContentMemory, leafNodeMemory);
    }
};

__device__ MemorySizes d_memorySizes;
MemorySizes h_memorySizes;

void copyMemorySizesToDevice(cudaStream_t stream)
{
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(d_memorySizes, &h_memorySizes, sizeof(h_memorySizes), 0, cudaMemcpyHostToDevice, stream));
}

template<typename T>
T getDeviceValue(T& deviceSymbol)
{
    T v;
    cudaMemcpyFromSymbol(&v, deviceSymbol, sizeof(T));
    return v;
}

template<typename T, CTuint size>
void getDeviceArray(T& deviceSymbol, T v[size])
{
    cudaMemcpyFromSymbol(v, deviceSymbol, size * sizeof(T));
}

template<typename T, typename A>
void cpyHostToDeviceVar(A& deviceSymbol, T v, cudaStream_t stream, CTuint offset = 0)
{
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(deviceSymbol, &v, sizeof(T), sizeof(T) * offset, cudaMemcpyHostToDevice, stream));
}

#define RESIZE_AND_CPY_PTR(__BUFFER, __SIZE, __STREAM) \
    m_##__BUFFER.Resize(__SIZE); \
    cpyHostToDeviceVar(d_##__BUFFER, m_##__BUFFER.GetPointer(), __STREAM);

#define RESIZE_AND_CPY_PTR_ARRAY(__BUFFER, __SIZE, __STREAM) \
    m_##__BUFFER.Resize(__SIZE); \
    cpyHostToDeviceVar(d_##__BUFFER, m_##__BUFFER[0].GetPointer(), __STREAM); \
    cpyHostToDeviceVar(d_##__BUFFER, m_##__BUFFER[1].GetPointer(), __STREAM, 1);

#define LOCATION __device__

LOCATION BBox*   d_nodes_BBox[2];

LOCATION CTuint* d_leafCountScanned;

LOCATION CTnodeIsLeaf_t* d_activeNodesIsLeaf;
LOCATION CTuint* d_activeNodes;
LOCATION CTuint* d_activeNodesThisLevel;

LOCATION CTuint* d_newActiveNodes;
LOCATION CTuint* d_newNodesContentCount;
LOCATION CTuint* d_newNodesContentStartAdd;

LOCATION CTeventIsLeaf_t* d_eventIsLeaf;
LOCATION CTuint* d_maskedInteriorContent;
LOCATION CTuint* d_scannedInteriorContent;

LOCATION CTuint* d_eventIsLeafScanned;
LOCATION CTuint* d_eventIsLeafScannedSums;

LOCATION CTuint* d_leafNodesContent;
LOCATION CTuint* d_leafNodesContentCount;
LOCATION CTuint* d_leafNodesContentStart;

LOCATION CTuint* d_scannedEventTypeEndMaskSums[3];
LOCATION CTuint* d_scannedClipSums[3];

CTuint* h_eventNodeIndex[2];

void dpEventLine::Resize(size_t newSize)
{
    if(indexedEvent.Size() < newSize)
    {
        indexedEvent.Resize(newSize);
        type.Resize(newSize);
        primId.Resize(newSize);
        ranges.Resize(newSize);
        typeStartScanned.Resize(newSize);
        scannedEventTypeEndMask.Resize(newSize);
        scannedEventTypeEndMaskSums.Resize(newSize);
        mask.Resize(newSize);
    }
}

void dpEventLines::Resize(size_t newSize, cudaStream_t pStream)
{

    //if(eventLines[0].mask.Size() >= newSize) return;

    for(int i = 0; i < 3; ++i)
    {
        eventLines[i].Resize((CTuint)(1.2 * newSize));
    }

    cuEventLineTriple src;
    src.lines[0] = eventLines[0].GetPtr(0);
    src.lines[1] = eventLines[1].GetPtr(0);
    src.lines[2] = eventLines[2].GetPtr(0);

    cuEventLineTriple dst;
    dst.lines[0] = eventLines[0].GetPtr(1);
    dst.lines[1] = eventLines[1].GetPtr(1);
    dst.lines[2] = eventLines[2].GetPtr(1);

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(d_scannedEventTypeEndMaskSums, eventLines[0].scannedEventTypeEndMaskSums.GetRawPointer(), sizeof(CTuint*), 0, cudaMemcpyHostToDevice, pStream));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(d_scannedEventTypeEndMaskSums, eventLines[1].scannedEventTypeEndMaskSums.GetRawPointer(), sizeof(CTuint*), 1 * sizeof(CTuint*), cudaMemcpyHostToDevice, pStream));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(d_scannedEventTypeEndMaskSums, eventLines[2].scannedEventTypeEndMaskSums.GetRawPointer(), sizeof(CTuint*), 2 * sizeof(CTuint*), cudaMemcpyHostToDevice, pStream));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(g_eventTriples, &src, sizeof(cuEventLineTriple), 0, cudaMemcpyHostToDevice, pStream));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(g_eventTriples, &dst, sizeof(cuEventLineTriple), sizeof(cuEventLineTriple), cudaMemcpyHostToDevice, pStream));
}

cuEventLine dpEventLine::GetPtr(CTbyte index)
{
    cuEventLine line;
    line.indexedEvent = indexedEvent[index].GetPointer();
    line.type = type[index].GetPointer();
    line.nodeIndex = h_eventNodeIndex[index];
    line.primId = primId[index].GetPointer();
    line.ranges = ranges[index].GetPointer();
    line.mask = mask.GetPointer();
    line.scannedEventTypeEndMask = typeStartScanned.GetConstPointer();

    return line;
}

void dpClipMask::Resize(size_t newSize, cudaStream_t pStream)
{
    if(mask0.Size() < newSize)
    {
        mask0.Resize(newSize);
        mask1.Resize(newSize);
        mask2.Resize(newSize);

        scannedMask0.Resize(newSize);
        scannedMask1.Resize(newSize);
        scannedMask2.Resize(newSize);

        scannedSums0.Resize(newSize);
        scannedSums1.Resize(newSize);
        scannedSums2.Resize(newSize);

        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(d_scannedClipSums, scannedSums0.GetRawPointer(), sizeof(CTuint*), 0, cudaMemcpyHostToDevice, pStream));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(d_scannedClipSums, scannedSums1.GetRawPointer(), sizeof(CTuint*), 1 * sizeof(CTuint*), cudaMemcpyHostToDevice, pStream));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(d_scannedClipSums, scannedSums2.GetRawPointer(), sizeof(CTuint*), 2 * sizeof(CTuint*), cudaMemcpyHostToDevice, pStream));

        newSplits0.Resize(newSize);
        newSplits1.Resize(newSize);
        newSplits2.Resize(newSize);

        index0.Resize(newSize);
        index1.Resize(newSize);
        index2.Resize(newSize);

        cuClipMaskArray mm;

        mm.mask[0].mask = mask0.GetPointer();
        mm.mask[1].mask = mask1.GetPointer();
        mm.mask[2].mask = mask2.GetPointer();

        mm.mask[0].newSplit = newSplits0.GetPointer();
        mm.mask[1].newSplit = newSplits1.GetPointer();
        mm.mask[2].newSplit = newSplits2.GetPointer();

        mm.mask[0].index = index0.GetPointer();
        mm.mask[1].index = index1.GetPointer();
        mm.mask[2].index = index2.GetPointer();

        mm.scanned[0] = scannedMask0.GetPointer();
        mm.scanned[1] = scannedMask1.GetPointer();
        mm.scanned[2] = scannedMask2.GetPointer();

        cuConstClipMask cmss[3];
        cmss[0].mask = mask0.GetConstPointer();
        cmss[1].mask = mask1.GetConstPointer();
        cmss[2].mask = mask2.GetConstPointer();

        cmss[0].newSplit = newSplits0.GetConstPointer();
        cmss[1].newSplit = newSplits1.GetConstPointer();
        cmss[2].newSplit = newSplits2.GetConstPointer();

        cmss[0].index = index0.GetConstPointer();
        cmss[1].index = index1.GetConstPointer();
        cmss[2].index = index2.GetConstPointer();

        cmss[0].scanned = scannedMask0.GetConstPointer();
        cmss[1].scanned = scannedMask1.GetConstPointer();
        cmss[2].scanned = scannedMask2.GetConstPointer();

        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(g_clipArray, &mm, sizeof(cuClipMaskArray), 0, cudaMemcpyHostToDevice, pStream));

        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(cms, &cmss, 3 * sizeof(cuConstClipMask), 0, cudaMemcpyHostToDevice, pStream));
    }
}
#undef DEBUG_CUDA
template <typename Operator, typename T>
__device__ void dpScanBinaryTriples(ConstTuple<3, T>& src, Tuple<3, CTuint>& scanned, Tuple<3, CTuint>& sums, CTuint N, Operator op)
{
#ifndef DEBUG_CUDA
    const CTuint block = 256;

    ConstTuple<3, CTuint> constSums;
    constSums.ts[0] = sums.ts[0];
    constSums.ts[1] = sums.ts[1];
    constSums.ts[2] = sums.ts[2];

    CTuint grid = nutty::cuda::GetCudaGrid(N, block);

    binaryTripleGroupScan<block><<<grid, block>>>(
        src, scanned, sums, op,
        N);

    CTuint sumsCount = nutty::cuda::GetCudaGrid(N, block);

    if(sumsCount > 1)
    {
        nutty::PrefixSumOp<CTuint> _op;
        completeScan2<256, 3><<<3, 256>>>(constSums, sums, _op, sumsCount);

        spreadScannedSums<<<grid-1, block>>>(scanned, sums, N);
    }
#endif
}

__device__ void dpScanClipMaskTriples(CTuint eventCount)
{
    ConstTuple<3, CTclipMask_t> ptr;
    ptr.ts[0] = g_clipArray.mask[0].mask;
    ptr.ts[1] = g_clipArray.mask[1].mask;
    ptr.ts[2] = g_clipArray.mask[2].mask;

    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = g_clipArray.scanned[0];
    ptr1.ts[1] = g_clipArray.scanned[1];
    ptr1.ts[2] = g_clipArray.scanned[2];
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = d_scannedClipSums[0];//g_clipArray.scannedSums[0];
    sums.ts[1] = d_scannedClipSums[1];
    sums.ts[2] = d_scannedClipSums[2];

    ClipMaskPrefixSumOP op;
    dpScanBinaryTriples(ptr, ptr1, sums, eventCount, op);
}

__device__ void dpScanEventTypesTriples(CTuint eventCount)
{
    CTuint eventBlock = 256U;
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    ConstTuple<3, CTeventType_t> ptr;
    ptr.ts[0] = g_eventTriples[d_toggleIndex].lines[0].type; //dpEventLines[0].type[toggleIndex].GetConstPointer();
    ptr.ts[1] = g_eventTriples[d_toggleIndex].lines[1].type; 
    ptr.ts[2] = g_eventTriples[d_toggleIndex].lines[2].type; 

    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = (CTuint*)g_eventTriples[d_toggleIndex].lines[0].scannedEventTypeEndMask; //dpEventLines[0].typeStartScanned.GetPointer();
    ptr1.ts[1] = (CTuint*)g_eventTriples[d_toggleIndex].lines[1].scannedEventTypeEndMask;
    ptr1.ts[2] = (CTuint*)g_eventTriples[d_toggleIndex].lines[2].scannedEventTypeEndMask;
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = d_scannedEventTypeEndMaskSums[0]; //dpEventLines[0].scannedEventTypeEndMaskSums.GetPointer();
    sums.ts[1] = d_scannedEventTypeEndMaskSums[1]; //dpEventLines[1].scannedEventTypeEndMaskSums.GetPointer();
    sums.ts[2] = d_scannedEventTypeEndMaskSums[2]; //dpEventLines[2].scannedEventTypeEndMaskSums.GetPointer();

    nutty::PrefixSumOp<CTuint> op;

    dpScanBinaryTriples(ptr, ptr1, sums, eventCount, op); 
}

__device__ void toggle(void)
{
    if(threadIdx.x == 0)
    {
        d_toggleIndex ^= 1;
    }
    
    __syncthreads();
}

template <
    CTuint blockSize, 
    typename Operator, 
    typename T
>
__device__ void completeBlockScan(const T* __restrict g_data, CTuint* scanned, CTuint* shrdMem, Operator op, CTuint N)
{  
    __shared__ CTuint prefixSum;
    prefixSum = 0;

    CTuint elem = op(op.GetNeutral());

    if(threadIdx.x < N)
    {
        elem = op(g_data[threadIdx.x]);
    }
      
    T nextElem = op.GetNeutral();

    for(CTuint offset = 0; offset < N; offset += blockSize)
    {
        uint gpos = offset + threadIdx.x;

        if(blockSize + gpos < N)
        {
            nextElem = g_data[blockSize + gpos];
        }

        CTuint sum = blockScan<blockSize>(shrdMem, elem);
        //CTuint sum = blockPrefixSums(shrdMem, elem);
        if(gpos < N)
        {
            scanned[gpos] = sum + prefixSum - elem;
        }

        __syncthreads();

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum;
        }

        elem = op(nextElem);

        //__syncthreads();
    }
}

#ifndef DEBUG_CUDA
__device__ void dpCcomputeSAH_Splits(CTuint nodeCount, CTuint eventCount, CTuint d = 0)
{
    dpScanEventTypesTriples(eventCount);

    const CTuint elemsPerThread = 1;
    CTuint sahBlock = 256U;
    CTuint sahGrid = nutty::cuda::GetCudaGrid(eventCount, sahBlock);

    computeSAHSplits3<1, elemsPerThread><<<sahGrid, sahBlock>>>(
        g_nodes.contentCount,
        g_nodes.contentStart,
        d_nodes_BBox[0],
        eventCount,
        d_toggleIndex);

    const CTuint blockSize = 512U;
    CTuint N = nodeCount * blockSize;
    segReduce<blockSize><<<nodeCount, blockSize>>>(g_splits.indexedSplit, N, eventCount);
}
#endif

template<CTuint BLOCK_SIZE>
__global__ void dpCreateKDTreeOld(CTuint primitiveCount, CTuint maxDepth)
{
#ifndef DEBUG_CUDA
    CTuint nodesCount = 1;
    CTuint eventCount = 2 * primitiveCount;

    CTuint interiorNodesCountOnThisLevel = 1;
    CTuint currentInteriorNodesCount = 1;
    CTuint nodeOffset = 0;
    CTuint leafContentOffset = 0;
    CTuint currentLeafCount = 0;
    CTuint childNodeOffset = 1;
    CTuint leafPrimitiveCount = 0;

#ifdef MEM_CHECK
    __shared__ bool breakOut;
    if(threadIdx.x == 0)
    {
        memset(d_needMoreMemory, 0, sizeof(d_needMoreMemory));
        breakOut = 0;
    }
    __syncthreads();
#endif

    __shared__ uint shrdMem[BLOCK_SIZE];

    for(CTuint d = 0; d < maxDepth; ++d)
    {
        CTuint block;
        CTuint grid;
        CTuint count;

        nodesCount = interiorNodesCountOnThisLevel;
        leafPrimitiveCount = 0;

        count = eventCount;
        block = 256U;
        grid = nutty::cuda::GetCudaGrid(count, block);

        if(threadIdx.x == 0)
        {

            dpCcomputeSAH_Splits(nodesCount, eventCount);

            createClipMask<<<grid, block>>>(
                g_nodes.contentStart, 
                g_nodes.contentCount,
                count,
                d_toggleIndex);

            dpScanClipMaskTriples(2 * count);

            CTuint _block = 256U;
            CTuint _grid = nutty::cuda::GetCudaGrid(2 * count, block);

            compactEventLineV2<<<_grid, _block>>>(
                2 * count, d_toggleIndex);

            cudaDeviceSynchronize();
        }

        toggle();

        CTuint childCount = 2 * nodesCount;

        CTbyte makeLeaves = d+1 == maxDepth;

        for(int offset = 0; offset < nodesCount; offset += blockDim.x)
        {
            CTuint id = offset + threadIdx.x;
            if(id < nodesCount)
            {
                CTuint edgesBeforeMe = 2 * g_nodes.contentStart[id];

                IndexedSAHSplit split = g_splitsConst.indexedSplit[edgesBeforeMe];

                CTaxis_t axis = g_splitsConst.axis[split.index];
                CTreal s = g_splitsConst.v[split.index];

                CTuint below = g_splitsConst.below[split.index];
                CTuint above = g_splitsConst.above[split.index];

                CTuint nodeId = nodeOffset + d_activeNodes[id];

                g_nodes.split[nodeId] = s;
                g_nodes.splitAxis[nodeId] = axis;

                CTuint dst = id;

                g_nodes.contentCount[2 * dst + 0] = below;
                g_nodes.contentCount[2 * dst + 1] = above;
                /*d_newNodesContentCount[2 * dst + 0] = below;
                d_newNodesContentCount[2 * dst + 1] = above;*/

                CTuint leftChildIndex = childNodeOffset + 2 * id + 0;
                CTuint rightChildIndex = childNodeOffset + 2 * id + 1;

                g_nodes.leftChild[nodeId] = leftChildIndex;
                g_nodes.rightChild[nodeId] = rightChildIndex;

                g_nodes.isLeaf[childNodeOffset + 2 * id + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                g_nodes.isLeaf[childNodeOffset + 2 * id + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;

                d_activeNodesIsLeaf[nodesCount + 2 * id + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                d_activeNodesIsLeaf[nodesCount + 2 * id + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                
                d_newActiveNodes[2 * id + 0] = 2 * id + 0;
                d_newActiveNodes[2 * id + 1] = 2 * id + 1;

                BBox l;
                BBox r;

                splitAABB(&d_nodes_BBox[0][id], s, axis, &l, &r);

                if(below > MAX_ELEMENTS_PER_LEAF)
                {
                    d_nodes_BBox[1][2 * dst + 0] = l;
                }

                if(above > MAX_ELEMENTS_PER_LEAF)
                {
                    d_nodes_BBox[1][2 * dst + 1] = r;
                }
            }
        }

        __syncthreads();

        for(int offset = 0; offset < childCount; offset += blockDim.x)
        {
            CTuint id = offset + threadIdx.x; 
            if(id < childCount)
            {
                //g_nodes.contentCount[id] = d_newNodesContentCount[id];
                d_activeNodes[id] = d_newActiveNodes[id];
            }
        }

//         if(threadIdx.x == 0)
//         {
//             swap(d_activeNodes, d_newActiveNodes);
//         }

        __syncthreads();

        eventCount = g_clipArray.scanned[0][2 * count - 1] + isSet(g_clipArray.mask[0].mask[2 * count - 1]);

        if(threadIdx.x == 0)
        {
            CTuint eventBlock = 256U;
            CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
            
            setEventsBelongToLeafAndSetNodeIndex<<<eventGrid, eventBlock>>>(
                d_activeNodesIsLeaf + nodesCount,
                d_eventIsLeaf,
                g_nodes.nodeToLeafIndex + childNodeOffset,
                eventCount,
                2 * nodesCount,
                d_toggleIndex);

            cudaDeviceSynchronize();
        }

        completeBlockScan<BLOCK_SIZE>(
            g_nodes.contentCount,
            g_nodes.contentStart,
            shrdMem, nutty::PrefixSumOp<CTuint>(), 
            childCount);
#ifdef MEM_CHECK
        if(childCount > d_memorySizes.perLevelNodeMemory)
        {
            d_deviceError = 1;
            break;
        }
#endif
        __syncthreads();

        completeBlockScan<BLOCK_SIZE>(
                    d_activeNodesIsLeaf + nodesCount, 
                    d_leafCountScanned, 
                    shrdMem, TypeOp<CTnodeIsLeaf_t>(), 
                    childCount);

        CTuint leafCount = d_leafCountScanned[childCount - 1] + (CTuint)d_activeNodesIsLeaf[nodesCount + childCount - 1];

        if(leafCount)
        {
#ifdef MEM_CHECK
          if(childCount > d_memorySizes.perLevelNodeMemory)
            {
                d_deviceError = 2;
//                 msgBuffer[0] = childCount;
//                 msgBuffer[1] = d_memorySizes.perLevelNodeMemory;
                break;
            }
#endif
            __syncthreads();

            for(int offset = 0; offset < childCount; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < childCount)
                {
                    CTclipMask_t mask = d_activeNodesIsLeaf[nodesCount + id];
                    d_maskedInteriorContent[id] = /*(mask < 2) */ (1 ^ mask) * g_nodes.contentCount[id];
                }
            }

            completeBlockScan<BLOCK_SIZE>(
                        d_maskedInteriorContent, 
                        d_scannedInteriorContent, 
                        shrdMem, nutty::PrefixSumOp<CTuint>(), 
                        childCount);

            const CTuint eventBlock = EVENT_GROUP_SIZE;
            CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

#ifdef MEM_CHECK
            if(eventCount > 2 * d_memorySizes.eventMemory)
            {
                d_deviceError = 3;
                break;
            }
#endif
            __syncthreads();

            if(threadIdx.x == 0)
            {

                binaryGroupScan<256><<<eventGrid, eventBlock>>>(
                    d_eventIsLeaf, d_eventIsLeafScanned, d_eventIsLeafScannedSums, TypeOp<CTeventIsLeaf_t>(), eventCount);

                CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);
                if(sumsCount > 1)
                {
                    nutty::PrefixSumOp<CTuint> _op;
                    completeScan<256><<<1, 256>>>(d_eventIsLeafScannedSums, d_eventIsLeafScannedSums, _op, sumsCount);
                
                    spreadScannedSumsSingle<<<eventGrid-1, eventBlock>>>(
                            d_eventIsLeafScanned, d_eventIsLeafScannedSums, eventCount);
                }
                cudaDeviceSynchronize();
            }

            //CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);

            __syncthreads();
 
//             if(sumsCount > 1)
//             {
//                 nutty::PrefixSumOp<CTuint> _op;
//                 completeBlockScan<BLOCK_SIZE>(d_eventIsLeafScannedSums.GetConstPointer(), d_eventIsLeafScannedSums.GetPointer(), shrdMem, _op, sumsCount);
// 
//                 if(threadIdx.x == 0)
//                 {
//                     spreadScannedSumsSingle<<<eventGrid-1, eventBlock>>>(
//                             d_eventIsLeafScanned.GetPointer(), d_eventIsLeafScannedSums.GetConstPointer(), eventCount);
//                     cudaDeviceSynchronize();
//                 }
//             }


#ifdef MEM_CHECK
            if(leafContentOffset + eventCount/2 > d_memorySizes.leafContentMemory || currentLeafCount + leafCount > d_memorySizes.leafNodeMemory)
            {
                d_deviceError = 4;
//                 msgBuffer[0] = leafContentOffset + eventCount/2 > d_memorySizes.leafContentMemory;
//                 msgBuffer[1] = currentLeafCount + leafCount > d_memorySizes.leafNodeMemory;
                break;
            }
#endif

            __syncthreads();

            if(threadIdx.x == 0)
            {
                compactMakeLeavesData<0><<<eventGrid, eventBlock>>>(
                    d_activeNodesIsLeaf + nodesCount,

                    g_nodes.contentStart,

                    d_eventIsLeafScanned,

                    g_nodes.contentCount,
                    d_eventIsLeaf,

                    d_leafCountScanned,

                    d_activeNodes,
                    d_leafCountScanned,
                    d_scannedInteriorContent,
                    d_nodes_BBox[1],

                    d_leafNodesContent,
                    g_nodes.nodeToLeafIndex,
                    d_newNodesContentCount,
                    d_newNodesContentStartAdd,
                    d_leafNodesContentStart,
                    d_leafNodesContentCount,
                    d_newActiveNodes,
                    d_nodes_BBox[0],

                    childNodeOffset,
                    leafContentOffset,
                    currentLeafCount,
                    childCount,
                    d_toggleIndex,
                    eventCount, d);

                cudaDeviceSynchronize();
            }

            for(int offset = 0; offset < childCount; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < childCount)
                {
                    CTnodeIsLeaf_t leafMask = d_activeNodesIsLeaf[nodesCount + id];
                    CTuint dst = d_leafCountScanned[id];
                    if(leafMask)
                    {
                        d_leafNodesContentStart[currentLeafCount + dst] = leafContentOffset + (g_nodes.contentStart[id] - d_scannedInteriorContent[id]);
                        d_leafNodesContentCount[currentLeafCount + dst] = g_nodes.contentCount[id];
                        g_nodes.nodeToLeafIndex[d_activeNodes[id] + childNodeOffset] = currentLeafCount + d_leafCountScanned[id];
                    }
                    else
                    {
                        dst = id - dst;
                        d_newActiveNodes[dst] = d_activeNodes[id];
                        d_newNodesContentCount[dst] = g_nodes.contentCount[id];
                        d_newNodesContentStartAdd[dst] = d_scannedInteriorContent[id];
                        d_nodes_BBox[0][dst] = d_nodes_BBox[1][id];
                    }
                }
            }

            __syncthreads();

            CTuint interiorPrimCount = d_scannedInteriorContent[childCount-1] +
                g_nodes.contentCount[childCount-1] * (d_activeNodesIsLeaf[childCount + nodesCount - 1] == 0);

            leafPrimitiveCount = eventCount/2 - interiorPrimCount;

            eventCount = 2 * interiorPrimCount;

//             if(threadIdx.x == 0)
//             {
//                 swap(g_nodes.contentCount, d_newNodesContentCount);
//                 swap(g_nodes.contentStart, d_newNodesContentStartAdd);
//                 swap(d_activeNodes, d_newActiveNodes);
//             }
            CTuint copyDistance = childCount - leafCount;
            for(int offset = 0; offset < copyDistance; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < copyDistance)
                {
                    g_nodes.contentCount[id] = d_newNodesContentCount[id];
                    g_nodes.contentStart[id] = d_newNodesContentStartAdd[id];
                    d_activeNodes[id] = d_newActiveNodes[id];
                }
            }

            toggle();
        }

        interiorNodesCountOnThisLevel = 2 * nodesCount - leafCount;
        currentInteriorNodesCount += interiorNodesCountOnThisLevel;
        nodeOffset = childNodeOffset;
        childNodeOffset += 2 * (nodesCount);

        leafContentOffset += leafPrimitiveCount;
        
        currentLeafCount += leafCount;

        if(eventCount == 0 || interiorNodesCountOnThisLevel == 0) //all nodes are leaf nodes
        {
            break;
        }

        if(!leafCount)
        {
            //swap(d_nodes_BBox[0], d_nodes_BBox[1]);
            for(int offset = 0; offset < interiorNodesCountOnThisLevel; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < interiorNodesCountOnThisLevel)
                {
                    d_nodes_BBox[0][id] = d_nodes_BBox[1][id];
                }
            }
        }

#ifdef MEM_CHECK
        if(d < maxDepth-1 && threadIdx.x == 0) //are we not done?
        {
            //check if we need more memory

            needMoreEventMemory = d_memorySizes.eventMemory < eventCount;
            needMoreNodeMemory = d_memorySizes.nodeMemory < (childNodeOffset + 2 * interiorNodesCountOnThisLevel);
            needMorePerLevelNodeMemory = d_memorySizes.perLevelNodeMemory < interiorNodesCountOnThisLevel + 2 * interiorNodesCountOnThisLevel;
            needMoreLeafNodeMemory = d_memorySizes.leafNodeMemory < currentLeafCount + 2 * interiorNodesCountOnThisLevel;
            needMoreLeafContentMemory = d_memorySizes.leafContentMemory < leafContentOffset + eventCount;

            d_currentLeafCount = currentLeafCount;
            d_currentEventCount = eventCount;
            d_maxLeafCountNextLevel = 2 * interiorNodesCountOnThisLevel;
            d_leafContentOffset = leafContentOffset;
            d_interiorNodesCountThisLevel = interiorNodesCountOnThisLevel;
//             msgBuffer[0] = d;
//             msgBuffer[1] = 2 * eventCount;
            for(int i = 0; i < 4; ++i)
            {
                if(d_needMoreMemory[i])
                {
                    breakOut = 1;
                    break;
                }
            }
        }

        __syncthreads();

        if(breakOut)
        {
            break;
        }
#else
        __syncthreads();
#endif

    }

#endif
}

struct dpTreeData
{
    CTuint eventCount;
    CTuint interiorNodesCountOnThisLevel;
    CTuint currentInteriorNodesCount;
    CTuint nodeOffset;
    CTuint leafContentOffset;
    CTuint currentLeafCount;
    CTuint childNodeOffset;
    CTuint currentDepth;
};

struct dpPerThreadData
{
    IndexedSAHSplit split;


};

template<CTuint BLOCK_SIZE>
__global__ void dpCreateKDTree(
    CTuint maxDepth,
    CTuint eventCount,
    CTuint interiorNodesCountOnThisLevel,
    CTuint currentInteriorNodesCount,
    CTuint nodeOffset,
    CTuint leafContentOffset,
    CTuint currentLeafCount,
    CTuint childNodeOffset,
    CTuint currentDepth) //dpTreeData data)
{
#ifndef DEBUG_CUDA

#ifdef MEM_CHECK
    __shared__ bool breakOut;
    if(threadIdx.x == 0)
    {
        memset(d_needMoreMemory, 0, sizeof(d_needMoreMemory));
        breakOut = 0;
    }
    __syncthreads();
#endif

    __shared__ CTuint shrdMem[BLOCK_SIZE];

    for(CTuint d = currentDepth; d < maxDepth; ++d)
    {
        CTuint block;
        CTuint grid;
        CTuint count;

        CTuint nodesCount = interiorNodesCountOnThisLevel;
        CTuint leafPrimitiveCount = 0;

        count = eventCount;
        block = 256U;
        grid = nutty::cuda::GetCudaGrid(count, block);

        if(threadIdx.x == 0)
        {
            dpCcomputeSAH_Splits(nodesCount, eventCount);

            createClipMask<<<grid, block>>>(
                g_nodes.contentStart, 
                g_nodes.contentCount,
                count,
                d_toggleIndex);

            dpScanClipMaskTriples(2 * count);

            CTuint _block = 256U;
            CTuint _grid = nutty::cuda::GetCudaGrid(2 * count, block);

            compactEventLineV2<<<_grid, _block>>>(
                2 * count, d_toggleIndex);

            cudaDeviceSynchronize();
        }
        __syncthreads();

#ifdef MEM_CHECK
            CTuint t = g_clipArray.scanned[0][2 * count - 1] + isSet(g_clipArray.mask[0].mask[2 * count - 1]);
            if(threadIdx.x == 0)
            {
                msgBuffer[10*d+0] = count;
            msgBuffer[10*d+1] = (CTreal)g_clipArray.scanned[0][2 * count - 1];
            msgBuffer[10*d+2] = (CTreal)interiorNodesCountOnThisLevel;
            msgBuffer[10*d+3] = (CTreal)currentInteriorNodesCount;
            msgBuffer[10*d+4] = (CTreal)nodeOffset;
            msgBuffer[10*d+5] = (CTreal)0;
            msgBuffer[10*d+6] = (CTreal)0;
            msgBuffer[10*d+7] = (CTreal)leafPrimitiveCount;
            msgBuffer[10*d+8] = (CTreal)d;
            msgBuffer[10*d+9] = (CTreal)-1;
            }
        if(t < 1000)
        {
            d_deviceError = d;
            return;
        }

#endif

        toggle();

        CTuint childCount = 2 * nodesCount;

        CTbyte makeLeaves = d+1 == maxDepth;

        for(int offset = 0; offset < nodesCount; offset += blockDim.x)
        {
            CTuint id = offset + threadIdx.x;
            if(id < nodesCount)
            {
                CTuint edgesBeforeMe = 2 * g_nodes.contentStart[id];

                IndexedSAHSplit split = g_splitsConst.indexedSplit[edgesBeforeMe];

                CTaxis_t axis = g_splitsConst.axis[split.index];
                CTreal s = g_splitsConst.v[split.index];

                CTuint below = g_splitsConst.below[split.index];
                CTuint above = g_splitsConst.above[split.index];

                CTuint nodeId = nodeOffset + d_activeNodes[id];

                g_nodes.split[nodeId] = s;
                g_nodes.splitAxis[nodeId] = axis;

                CTuint dst = id;

                g_nodes.contentCount[2 * dst + 0] = below;
                g_nodes.contentCount[2 * dst + 1] = above;
                /*d_newNodesContentCount[2 * dst + 0] = below;
                d_newNodesContentCount[2 * dst + 1] = above;*/

                CTuint leftChildIndex = childNodeOffset + 2 * id + 0;
                CTuint rightChildIndex = childNodeOffset + 2 * id + 1;

                g_nodes.leftChild[nodeId] = leftChildIndex;
                g_nodes.rightChild[nodeId] = rightChildIndex;

                g_nodes.isLeaf[childNodeOffset + 2 * id + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                g_nodes.isLeaf[childNodeOffset + 2 * id + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;

                d_activeNodesIsLeaf[nodesCount + 2 * id + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                d_activeNodesIsLeaf[nodesCount + 2 * id + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                
                d_newActiveNodes[2 * id + 0] = 2 * id + 0;
                d_newActiveNodes[2 * id + 1] = 2 * id + 1;

                BBox l;
                BBox r;

                splitAABB(&d_nodes_BBox[0][id], s, axis, &l, &r);

                if(below > MAX_ELEMENTS_PER_LEAF)
                {
                    d_nodes_BBox[1][2 * dst + 0] = l;
                }

                if(above > MAX_ELEMENTS_PER_LEAF)
                {
                    d_nodes_BBox[1][2 * dst + 1] = r;
                }
            }
        }

        __syncthreads();
#ifdef MEM_CHECK
        for(int offset = 0; offset < childCount; offset += blockDim.x)
        {
            CTuint id = offset + threadIdx.x; 
            if(id < childCount)
            {
                //g_nodes.contentCount[id] = d_newNodesContentCount[id];
                d_activeNodes[id] = d_newActiveNodes[id];
            }
        }
#else
        if(threadIdx.x == 0)
        {
            swap(d_activeNodes, d_newActiveNodes);
        }
#endif

        __syncthreads();

        eventCount = g_clipArray.scanned[0][2 * count - 1] + isSet(g_clipArray.mask[0].mask[2 * count - 1]);

        if(threadIdx.x == 0)
        {
            CTuint eventBlock = 256U;
            CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
            
            setEventsBelongToLeafAndSetNodeIndex<<<eventGrid, eventBlock>>>(
                d_activeNodesIsLeaf + nodesCount,
                d_eventIsLeaf,
                g_nodes.nodeToLeafIndex + childNodeOffset,
                eventCount,
                2 * nodesCount,
                d_toggleIndex);

            cudaDeviceSynchronize();
        }

        completeBlockScan<BLOCK_SIZE>(
            g_nodes.contentCount,
            g_nodes.contentStart,
            shrdMem, nutty::PrefixSumOp<CTuint>(), 
            childCount);

#ifdef MEM_CHECK
        if(childCount > d_memorySizes.perLevelNodeMemory)
        {
            d_deviceError = 1;
            break;
        }
#endif
        __syncthreads();

        completeBlockScan<BLOCK_SIZE>(
                    d_activeNodesIsLeaf + nodesCount, 
                    d_leafCountScanned, 
                    shrdMem, TypeOp<CTnodeIsLeaf_t>(), 
                    childCount);

        CTuint leafCount = d_leafCountScanned[childCount - 1] + (CTuint)d_activeNodesIsLeaf[nodesCount + childCount - 1];
        CTuint interiorPrimCount;
        if(leafCount)
        {
#ifdef MEM_CHECK
          if(childCount > d_memorySizes.perLevelNodeMemory)
            {
                d_deviceError = 2;
//                 msgBuffer[0] = childCount;
//                 msgBuffer[1] = d_memorySizes.perLevelNodeMemory;
                break;
            }
#endif
            __syncthreads();

            for(int offset = 0; offset < childCount; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < childCount)
                {
                    CTclipMask_t mask = d_activeNodesIsLeaf[nodesCount + id];
                    d_maskedInteriorContent[id] = /*(mask < 2) */ (1 ^ mask) * g_nodes.contentCount[id];
                }
            }

            completeBlockScan<BLOCK_SIZE>(
                        d_maskedInteriorContent, 
                        d_scannedInteriorContent, 
                        shrdMem, nutty::PrefixSumOp<CTuint>(), 
                        childCount);

            const CTuint eventBlock = EVENT_GROUP_SIZE;
            CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

#ifdef MEM_CHECK
            if(eventCount > 2 * d_memorySizes.eventMemory)
            {
                d_deviceError = 3;
                break;
            }
#endif
            __syncthreads();

            if(threadIdx.x == 0)
            {

                binaryGroupScan<256><<<eventGrid, eventBlock>>>(
                    d_eventIsLeaf, d_eventIsLeafScanned, d_eventIsLeafScannedSums, TypeOp<CTeventIsLeaf_t>(), eventCount);

                CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);
                if(sumsCount > 1)
                {
                    nutty::PrefixSumOp<CTuint> _op;
                    completeScan<256><<<1, 256>>>(d_eventIsLeafScannedSums, d_eventIsLeafScannedSums, _op, sumsCount);
                
                    spreadScannedSumsSingle<<<eventGrid-1, eventBlock>>>(
                            d_eventIsLeafScanned, d_eventIsLeafScannedSums, eventCount);
                }
                cudaDeviceSynchronize();
            }

            //CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);

            __syncthreads();
 
//             if(sumsCount > 1)
//             {
//                 nutty::PrefixSumOp<CTuint> _op;
//                 completeBlockScan<BLOCK_SIZE>(d_eventIsLeafScannedSums.GetConstPointer(), d_eventIsLeafScannedSums.GetPointer(), shrdMem, _op, sumsCount);
// 
//                 if(threadIdx.x == 0)
//                 {
//                     spreadScannedSumsSingle<<<eventGrid-1, eventBlock>>>(
//                             d_eventIsLeafScanned.GetPointer(), d_eventIsLeafScannedSums.GetConstPointer(), eventCount);
//                     cudaDeviceSynchronize();
//                 }
//             }


#ifdef MEM_CHECK
            if(leafContentOffset + eventCount/2 > d_memorySizes.leafContentMemory || currentLeafCount + leafCount > d_memorySizes.leafNodeMemory)
            {
                d_deviceError = 4;
                msgBuffer[0] = leafContentOffset + eventCount/2;
                msgBuffer[1] = currentLeafCount + leafCount;
                break;
            }
#endif

            __syncthreads();

            if(threadIdx.x == 0)
            {
                compactMakeLeavesData<0><<<eventGrid, eventBlock>>>(
                    d_activeNodesIsLeaf + nodesCount,

                    g_nodes.contentStart,

                    d_eventIsLeafScanned,

                    g_nodes.contentCount,
                    d_eventIsLeaf,

                    d_leafCountScanned,

                    d_activeNodes,
                    d_leafCountScanned,
                    d_scannedInteriorContent,
                    d_nodes_BBox[1],

                    d_leafNodesContent,
                    g_nodes.nodeToLeafIndex,
                    d_newNodesContentCount,
                    d_newNodesContentStartAdd,
                    d_leafNodesContentStart,
                    d_leafNodesContentCount,
                    d_newActiveNodes,
                    d_nodes_BBox[0],

                    childNodeOffset,
                    leafContentOffset,
                    currentLeafCount,
                    childCount,
                    d_toggleIndex,
                    eventCount, d);

                cudaDeviceSynchronize();
            }

            for(int offset = 0; offset < childCount; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < childCount)
                {
                    CTnodeIsLeaf_t leafMask = d_activeNodesIsLeaf[nodesCount + id];
                    CTuint dst = d_leafCountScanned[id];
                    if(leafMask)
                    {
                        d_leafNodesContentStart[currentLeafCount + dst] = leafContentOffset + (g_nodes.contentStart[id] - d_scannedInteriorContent[id]);
                        d_leafNodesContentCount[currentLeafCount + dst] = g_nodes.contentCount[id];
                        g_nodes.nodeToLeafIndex[d_activeNodes[id] + childNodeOffset] = currentLeafCount + d_leafCountScanned[id];
                    }
                    else
                    {
                        dst = id - dst;
                        d_newActiveNodes[dst] = d_activeNodes[id];
                        d_newNodesContentCount[dst] = g_nodes.contentCount[id];
                        d_newNodesContentStartAdd[dst] = d_scannedInteriorContent[id];
                        d_nodes_BBox[0][dst] = d_nodes_BBox[1][id];
                    }
                }
            }

            __syncthreads();

            interiorPrimCount = d_scannedInteriorContent[childCount-1] +
                g_nodes.contentCount[childCount-1] * (d_activeNodesIsLeaf[childCount + nodesCount - 1] == 0);

            leafPrimitiveCount = eventCount/2 - interiorPrimCount;
            
            eventCount = 2 * interiorPrimCount;

#ifdef MEM_CHECK
            CTuint copyDistance = childCount - leafCount;
            for(int offset = 0; offset < copyDistance; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < copyDistance)
                {
                    g_nodes.contentCount[id] = d_newNodesContentCount[id];
                    g_nodes.contentStart[id] = d_newNodesContentStartAdd[id];
                    d_activeNodes[id] = d_newActiveNodes[id];
                }
            }
#else
            if(threadIdx.x == 0)
            {
                swap(g_nodes.contentCount, d_newNodesContentCount);
                swap(g_nodes.contentStart, d_newNodesContentStartAdd);
                swap(d_activeNodes, d_newActiveNodes);
            }
#endif

            toggle();
        }

        interiorNodesCountOnThisLevel = 2 * nodesCount - leafCount;
        currentInteriorNodesCount += interiorNodesCountOnThisLevel;
        nodeOffset = childNodeOffset;
        childNodeOffset += 2 * (nodesCount);

        leafContentOffset += leafPrimitiveCount;

        currentLeafCount += leafCount;

        if(eventCount == 0 || interiorNodesCountOnThisLevel == 0) //all nodes are leaf nodes
        {
            break;
        }

        if(!leafCount)
        {
#ifdef MEM_CHECK
            for(int offset = 0; offset < interiorNodesCountOnThisLevel; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < interiorNodesCountOnThisLevel)
                {
                    d_nodes_BBox[0][id] = d_nodes_BBox[1][id];
                }
            }
#else
            if(threadIdx.x == 0) swap(d_nodes_BBox[0], d_nodes_BBox[1]);
#endif
        }

#ifdef MEM_CHECK
        if(d < maxDepth-1 && threadIdx.x == 0) //are we not done?
        {
            //check if we need more memory

            needMoreEventMemory = d_memorySizes.eventMemory < eventCount;
            needMoreNodeMemory = d_memorySizes.nodeMemory < (childNodeOffset + 2 * interiorNodesCountOnThisLevel);
            needMorePerLevelNodeMemory = d_memorySizes.perLevelNodeMemory < interiorNodesCountOnThisLevel + 2 * interiorNodesCountOnThisLevel;
            needMoreLeafNodeMemory = d_memorySizes.leafNodeMemory < currentLeafCount + 2 * interiorNodesCountOnThisLevel;
            needMoreLeafContentMemory = d_memorySizes.leafContentMemory < leafContentOffset + eventCount;

            d_currentLeafCount = currentLeafCount;
            d_currentEventCount = eventCount;
            d_maxLeafCountNextLevel = 2 * interiorNodesCountOnThisLevel;
            d_leafContentOffset = leafContentOffset;
            d_interiorNodesCountThisLevel = interiorNodesCountOnThisLevel;
//             msgBuffer[0] = d;
//             msgBuffer[1] = 2 * data.eventCount;
            for(int i = 0; i < 4; ++i)
            {
                if(d_needMoreMemory[i])
                {
                    breakOut = 1;
                    break;
                }
            }
        }

        __syncthreads();

        if(breakOut)
        {
            break;
        }
#else
        __syncthreads();
#endif

//         if(1
// //             && (d < maxDepth-1) &&  
// //             data.interiorNodesCountOnThisLevel > 64  && 
// //             data.interiorNodesCountOnThisLevel < 1024 && 
// //             (data.interiorNodesCountOnThisLevel > (BLOCK_SIZE<<1) || data.interiorNodesCountOnThisLevel < (BLOCK_SIZE>>1)))
//     )
//         {

            if(threadIdx.x == 0)
            {
               // msgBuffer[d+0] = childNodeOffset;
                //currentDepth = d+1;

                dpCreateKDTree<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(









                    maxDepth,
                    eventCount,
                    interiorNodesCountOnThisLevel,
                    currentInteriorNodesCount,
                    nodeOffset,
                    leafContentOffset,
                    currentLeafCount,
                    childNodeOffset,
                    currentDepth+1);

                /*
                CTuint newblockSize;
//                 if(nutty::Ispow2((int)data.interiorNodesCountOnThisLevel))
//                 {
//                     newblockSize = data.interiorNodesCountOnThisLevel;
//                 }
//                 else
//                 {
//                     newblockSize = 1 << (32 - __clz(data.interiorNodesCountOnThisLevel) - 1);
//                 }

                newblockSize = 64;//min(1024, max(newblockSize, 64));
    // 
                //data.currentDepth = d+1;
                msgBuffer[d+0] = (CTreal)nodesCount;
//                 msgBuffer[4*d+1] = (CTreal)nodesCount;
//                 msgBuffer[4*d+2] = (CTreal)BLOCK_SIZE;
//                 msgBuffer[4*d+3] = (CTreal)newblockSize;
                
                //d_deviceError = d;
#if 1
                cudaStream_t pStream;
                cudaStreamCreateWithFlags(&pStream, cudaStreamNonBlocking);
                switch(newblockSize)
                {
                    case 64  : 
    //                 case 128 : dpCreateKDTree<128><<<1,  128>>>(maxDepth, data); break;
    //                 case 256 : dpCreateKDTree<256><<<1,  256>>>(maxDepth, data); break;
    //                 case 512 : dpCreateKDTree<512><<<1,  512>>>(maxDepth, data); break;
    //                 case 1024: dpCreateKDTree<1024><<<1, 1024>>>(maxDepth, data); break;
                }
#endif
                */
          //  }
        }

//         __syncthreads();
        return;
    }
#endif
}

__global__ void dpInit(CTuint primitiveCount, const BBox* __restrict sceneBBox, const BBox* __restrict primBBox, CTbyte initMemory)
{
    if(threadIdx.x == 0)
    {
        d_toggleIndex = 0;

        CTuint elementBlock = 256U;
        CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);

        g_nodes.contentStart[0] = 0;
        g_nodes.contentCount[0] = primitiveCount;
#ifndef DEBUG_CUDA
        createEventsAndInit3<1, 0><<<elementGrid, elementBlock>>>(
            primBBox, 
            sceneBBox,

            d_activeNodes,
            g_nodes.nodeToLeafIndex,
            g_nodes.isLeaf,
            g_nodes.contentCount,
            d_nodes_BBox[0],

            primitiveCount);

        for(CTaxis_t i = 0; i < 3; ++i)
        {
            nutty::Sort(
                nutty::DevicePtr_Cast<IndexedEvent>(g_eventTriples[0].lines[i].indexedEvent), 
                nutty::DevicePtr_Cast<IndexedEvent>(g_eventTriples[0].lines[i].indexedEvent + 2 * primitiveCount), 
                EventSort());
        }
        
        reorderEvent3<<<2 * elementGrid, elementBlock>>>(2 * primitiveCount);
#endif
        d_toggleIndex = 1;
        
        d_mem_alloc_error = 0;
        d_deviceError = 0;
    }
}

CT_RESULT cudpKDTree::Update(void)
{
    CTuint primitiveCount = (CTuint)(m_currentTransformedVertices.Size() / 3);

    if(!m_initialized)
    {
        size_t limit;
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));

#ifdef MEM_CHECK
        GrowPerLevelNodeMemory(1*2048);
        GrowNodeMemory(1*2048);
        GrowEventMemory(2 * primitiveCount);
        GrowLeafNodeMemory(1*2048);
        GrowLeafContentMemory(2 * primitiveCount);
#else
        //dragon
        GrowPerLevelNodeMemory(19384);
        GrowNodeMemory(131072);
        GrowEventMemory(3504132);
        GrowLeafNodeMemory(41434);
        GrowLeafContentMemory(3504132);

        //bunny
        GrowPerLevelNodeMemory(4*2048);
        GrowNodeMemory(4*2048);
        GrowEventMemory(4 * primitiveCount);
        GrowLeafNodeMemory(8*2048);
        GrowLeafContentMemory(4 * primitiveCount);
#endif

        m_primAABBs.Resize(primitiveCount);
        m_sceneBBox.Resize(m_primAABBs.Size()/2);
    }

    if(!m_initialized)
    {
        m_initialized = true;
        m_depth = (byte)min(64, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));
    }

    //static bool staticc = true;
    DEVICE_SYNC_CHECK();
    cudaCreateTriangleAABBs(m_currentTransformedVertices.GetPointer(), m_primAABBs.GetPointer(), primitiveCount, m_pStream);
    DEVICE_SYNC_CHECK();
   // if(staticc)
    {
        DEVICE_SYNC_CHECK();

        static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
        static float3 min3f = -max3f;

        BBox bboxN;
        bboxN.m_min = max3f; 
        bboxN.m_max = min3f;
        nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), ReduceBBox(), bboxN, m_pStream);
      //  staticc = false;
    }

    DEVICE_SYNC_CHECK();
    bool done = false;
#ifdef MEM_CHECK
    int t = 1;
#endif

    while(!done)
    {
#ifdef MEM_CHECK
        CTuint h_needMoreMemory[5] = {0,0,0,0,0};
        __ct_printf("Try: %d\n", t++);
#endif


//         CTuint elementBlock = 256U;
//         CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);
// 
//         createEventsAndInit3<1, 0><<<elementGrid, elementBlock, 0, m_pStream>>>(
//             m_primAABBs.GetConstPointer(), 
//             m_sceneBBox.GetConstPointer(),
// 
//             m_activeNodes.GetPointer(),
//             m_nodes_NodeIdToLeafIndex.GetPointer(),
//             m_nodes_IsLeaf.GetPointer(),
//             m_nodes_ContentCount.GetPointer(),
//             m_nodes_BBox[0].GetPointer(),
// 
//             primitiveCount);
// // 
//         DEVICE_SYNC_CHECK();
// 
//         for(CTbyte i = 0; i < 3; ++i)
//         {
//             nutty::Sort(
//                 nutty::DevicePtr_Cast<IndexedEvent>(m_eventLines.eventLines[i].GetPtr(0).indexedEvent), 
//                 nutty::DevicePtr_Cast<IndexedEvent>(m_eventLines.eventLines[i].GetPtr(0).indexedEvent + 2 * primitiveCount), 
//                 EventSort(),
//                 m_pStream);
//         }
//         
//         reorderEvent3<<<2 * elementGrid, elementBlock, 0, m_pStream>>>(2 * primitiveCount);
//         DEVICE_SYNC_CHECK();
        dpInit<<<1, 1, 0, m_pStream>>>(primitiveCount, m_sceneBBox.GetPointer(), m_primAABBs.GetPointer(), !m_initialized);

        dpTreeData data;
        data.eventCount = 2 * primitiveCount;
        data.interiorNodesCountOnThisLevel = 1;
        data.currentInteriorNodesCount = 1;
        data.nodeOffset = 0;
        data.currentLeafCount = 0;
        data.childNodeOffset = 1;
        data.leafContentOffset = 0;
        data.currentDepth = 0;

        const CTuint BLOCK_SIZE = 64;
        dpCreateKDTree<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, m_pStream>>>(m_depth, data.eventCount, 1,1,0,0,0,1,0);
        
#ifdef MEM_CHECK
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyFromSymbol(h_needMoreMemory, d_needMoreMemory, sizeof(h_needMoreMemory)));

        CTuint h_error = getDeviceValue(d_deviceError);
        CTreal h_msgBuffer[256];
        cudaMemcpyFromSymbol(&h_msgBuffer, msgBuffer, sizeof(h_msgBuffer));
        bool needMoreMemory = false;
        for(int i = 0; i < 5; ++i)
        {
            //__ct_printf("%d ", h_needMoreMemory[i]);
            needMoreMemory |= h_needMoreMemory[i];
        }
        __ct_printf("\n");
        for(int i = 0; i < sizeof(h_msgBuffer)/4; ++i)
        {
            if(h_msgBuffer[i] == -1) __ct_printf("\n");
            else __ct_printf("%.0f ", h_msgBuffer[i]);
        }
//         __ct_printf("\n");
        done = true;

        if(h_error)
        {
            __ct_printf("\nd_deviceError=%d\n", h_error);
            //DEVICE_SYNC_CHECK();
            exit(0);
        }

        if(needMoreMemory)
        {
            __ct_printf("Need more Memory: (%d %d %d %d %d) d=%f\n", h_needMoreMemory[0], h_needMoreMemory[1], h_needMoreMemory[2], h_needMoreMemory[3], h_needMoreMemory[4], h_msgBuffer[0]);
        }

        if(h_needMoreEventMemory)
        {
            GrowEventMemory();
            done = false;
        }

        if(h_needMoreNodeMemory)
        {
            GrowNodeMemory();
            done = false;
        }

        if(h_needMorePerLevelNodeMemory)
        {
            CTuint interiorNodesThisLevel = getDeviceValue(d_interiorNodesCountThisLevel);
            GrowPerLevelNodeMemory(4 * 2 * interiorNodesThisLevel);
            done = false;
        }

        if(h_needMoreLeafNodeMemory)
        {
            GrowLeafNodeMemory();
            done = false;
        }

        if(h_needMoreLeafContentMemory)
        {
            GrowLeafContentMemory();
            done = false;
        }


        DEVICE_SYNC_CHECK();
        h_memorySizes.print();
#else
        done = true;
#endif
    }

//     CTreal h_msgBuffer[256];
//     cudaMemcpyFromSymbol(&h_msgBuffer, msgBuffer, sizeof(h_msgBuffer));
// 
//     for(int i = 0; i < sizeof(h_msgBuffer)/4; ++i)
//     {
//         __ct_printf("%.0f ", h_msgBuffer[i]);
//     }
// 
//     __ct_printf("\n");


//     ct_printf("Tree Summary:\n");
//     PRINT_BUFFER(m_nodes_IsLeaf);
//     PRINT_BUFFER(m_nodes_Split);
//     PRINT_BUFFER(m_nodes_SplitAxis);
//     PRINT_BUFFER(m_nodes_LeftChild);
//     PRINT_BUFFER(m_nodes_RightChild);
//     PRINT_BUFFER(m_leafNodesContentCount);
//     PRINT_BUFFER(m_leafNodesContentStart);
//     PRINT_BUFFER(m_nodes_NodeIdToLeafIndex);
// 
//     if(m_leafNodesContent.Size() < 1024)
//     {
//         PRINT_BUFFER(m_leafNodesContent);
//     }

    DEVICE_SYNC_CHECK();
    return CT_SUCCESS;
}

void cudpKDTree::GrowPerLevelNodeMemory(CTuint newSize)
{
    h_memorySizes.perLevelNodeMemory = newSize;
    copyMemorySizesToDevice(m_pStream);
    
    //m_leafCountScanned.Resize(newSize);
    RESIZE_AND_CPY_PTR(leafCountScanned, newSize, m_pStream);

    RESIZE_AND_CPY_PTR(maskedInteriorContent, newSize, m_pStream);
    RESIZE_AND_CPY_PTR(scannedInteriorContent, newSize, m_pStream);

    RESIZE_AND_CPY_PTR(activeNodesIsLeaf, newSize, m_pStream);
    RESIZE_AND_CPY_PTR(activeNodes, newSize, m_pStream);
    RESIZE_AND_CPY_PTR(activeNodesThisLevel, newSize, m_pStream);
    RESIZE_AND_CPY_PTR(newActiveNodes, newSize, m_pStream);
    RESIZE_AND_CPY_PTR_ARRAY(nodes_BBox, newSize, m_pStream);

    m_nodes_ContentCount.Resize(newSize);
    m_nodes_ContentStartAdd.Resize(newSize);
//     RESIZE_AND_CPY_PTR(nodes_ContentStartAdd, newSize, m_pStream);
//     RESIZE_AND_CPY_PTR(nodes_ContentCount, newSize, m_pStream);

    //m_newNodesContentCount.Resize(newSize);
    RESIZE_AND_CPY_PTR(newNodesContentCount, newSize, m_pStream);
    RESIZE_AND_CPY_PTR(newNodesContentStartAdd, newSize, m_pStream);
    //d_nodes_newContentCount

    m_nodes.isLeaf = m_nodes_IsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodes_SplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodes_Split.GetDevicePtr()();
    m_nodes.contentStart = m_nodes_ContentStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodes_ContentCount.GetDevicePtr()();
    m_nodes.leftChild = m_nodes_LeftChild.GetDevicePtr()();
    m_nodes.rightChild = m_nodes_RightChild.GetDevicePtr()();
    m_nodes.nodeToLeafIndex = m_nodes_NodeIdToLeafIndex.GetDevicePtr()();

    cpyHostToDeviceVar(g_nodes, m_nodes, m_pStream);
}

void cudpKDTree::GrowLeafNodeMemory(CTuint newSize)
{
    if(newSize == (CTuint)-1)
    {
        CTuint leafCount = getDeviceValue(d_maxLeafCountNextLevel);
        CTuint currentLeafCount = getDeviceValue(d_currentLeafCount);
        newSize = leafCount + currentLeafCount;
        __ct_printf("leafCount=%d currentLeafCount=%d\n\n", leafCount, currentLeafCount);
    }

    RESIZE_AND_CPY_PTR(leafNodesContentStart, newSize, m_pStream);
    RESIZE_AND_CPY_PTR(leafNodesContentCount, newSize, m_pStream);

    h_memorySizes.leafNodeMemory = (CTuint)m_leafNodesContentCount.Size();

    copyMemorySizesToDevice(m_pStream);
}

void cudpKDTree::GrowLeafContentMemory(CTuint newSize)
{
    if(newSize == (CTuint)-1)
    {
        CTuint leafContentOffset = getDeviceValue(d_leafContentOffset);
        CTuint eventCount = 2 * getDeviceValue(d_currentEventCount);
        newSize = eventCount + leafContentOffset;
    }

    RESIZE_AND_CPY_PTR(leafNodesContent, newSize, m_pStream);

    h_memorySizes.leafContentMemory = (CTuint)m_leafNodesContent.Size();

    copyMemorySizesToDevice(m_pStream);
}

void cudpKDTree::GrowNodeMemory(CTuint newSize)
{
    newSize = (CTuint)(m_nodes_IsLeaf.Size() ? m_nodes_IsLeaf.Size() * 4 : newSize);

    h_memorySizes.nodeMemory = newSize;
    copyMemorySizesToDevice(m_pStream);

//     RESIZE_AND_CPY_PTR(nodes_IsLeaf, newSize, m_pStream);
//     RESIZE_AND_CPY_PTR(nodes_Split, newSize, m_pStream);
//     RESIZE_AND_CPY_PTR(nodes_NodeIdToLeafIndex, newSize, m_pStream);
//     RESIZE_AND_CPY_PTR(nodes_SplitAxis, newSize, m_pStream);
//     RESIZE_AND_CPY_PTR(nodes_LeftChild, newSize, m_pStream);
//     RESIZE_AND_CPY_PTR(nodes_RightChild, newSize, m_pStream);

    m_nodes_IsLeaf.Resize(newSize);
    m_nodes_Split.Resize(newSize);
    m_nodes_NodeIdToLeafIndex.Resize(newSize);
    m_nodes_SplitAxis.Resize(newSize);
    m_nodes_LeftChild.Resize(newSize);
    m_nodes_RightChild.Resize(newSize);

    m_nodes.isLeaf = m_nodes_IsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodes_SplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodes_Split.GetDevicePtr()();
    m_nodes.contentStart = m_nodes_ContentStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodes_ContentCount.GetDevicePtr()();
    m_nodes.leftChild = m_nodes_LeftChild.GetDevicePtr()();
    m_nodes.rightChild = m_nodes_RightChild.GetDevicePtr()();
    m_nodes.nodeToLeafIndex = m_nodes_NodeIdToLeafIndex.GetDevicePtr()();

    cpyHostToDeviceVar(g_nodes, m_nodes, m_pStream);
}

void cudpKDTree::GrowEventMemory(CTuint eventCount)
{
    if(eventCount == (CTuint)-1)
    {
        eventCount = 2 * getDeviceValue(d_currentEventCount);
    }

    h_memorySizes.eventMemory = eventCount;
    copyMemorySizesToDevice(m_pStream);

    RESIZE_AND_CPY_PTR(eventIsLeafScanned, 2 * eventCount, m_pStream);
    RESIZE_AND_CPY_PTR(eventIsLeafScannedSums, (2 * eventCount) / 256 + 256, m_pStream);

//     RESIZE_AND_CPY_PTR(splits_Above, eventCount, m_pStream);
//     RESIZE_AND_CPY_PTR(splits_Below, eventCount, m_pStream);
//     RESIZE_AND_CPY_PTR(splits_Axis, eventCount, m_pStream);
//     RESIZE_AND_CPY_PTR(splits_Plane, eventCount, m_pStream);
//     RESIZE_AND_CPY_PTR(splits_IndexedSplit, eventCount, m_pStream);

    RESIZE_AND_CPY_PTR(eventIsLeaf, eventCount, m_pStream);

    m_splits_Above.Resize(eventCount);
    m_splits_Below.Resize(eventCount);
    m_splits_Axis.Resize(eventCount);
    m_splits_Plane.Resize(eventCount);
    m_splits_IndexedSplit.Resize(eventCount);

    //m_eventIsLeaf.Resize(eventCount);
    
    m_splits.above = m_splits_Above.GetDevicePtr()();
    m_splits.below = m_splits_Below.GetDevicePtr()();
    m_splits.axis = m_splits_Axis.GetDevicePtr()();
    m_splits.indexedSplit = m_splits_IndexedSplit.GetDevicePtr()();
    m_splits.v = m_splits_Plane.GetDevicePtr()();

    m_eventNodeIndex.Resize(eventCount);

    h_eventNodeIndex[0] = m_eventNodeIndex[0].GetPointer();
    h_eventNodeIndex[1] = m_eventNodeIndex[1].GetPointer();

    cpyHostToDeviceVar(g_splits, m_splits, m_pStream);
    
    SplitConst splitsConst;
    splitsConst.above = m_splits_Above.GetDevicePtr()();
    splitsConst.below = m_splits_Below.GetDevicePtr()();
    splitsConst.axis = m_splits_Axis.GetDevicePtr()();
    splitsConst.indexedSplit = m_splits_IndexedSplit.GetDevicePtr()();
    splitsConst.v = m_splits_Plane.GetDevicePtr()();

    cpyHostToDeviceVar(g_splitsConst, splitsConst, m_pStream);

    m_eventLines.Resize(2 * eventCount, m_pStream);
    m_clipsMask.Resize(2 * eventCount, m_pStream);
}

#else
CT_RESULT cudpKDTree::Update(void)
{
    return CT_NOT_YET_IMPLEMENTED;
}

void cudpKDTree::GrowEventMemory(CTuint eventCount)
{

}

void cudpKDTree::GrowNodeMemory(CTuint newSize)
{

}

void cudpKDTree::GrowLeafContentMemory(CTuint newSize)
{

}

void cudpKDTree::GrowPerLevelNodeMemory(CTuint newSize)
{

}
#endif

const void* cudpKDTree::GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const
{
    return cuKDTree::GetLinearMemory(type, byteCount);
}