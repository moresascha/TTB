#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#define NUTTY_DEBUG

#include "cuKDTree.h"

#define NODES_GROUP_SIZE 256U
#define EVENT_GROUP_SIZE 256U

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

__device__ CTuint deviceError;
__device__ CTuint mem_alloc_error;

__device__ CTreal msgBuffer[256];

__device__ int dstrlen(const char* text)
{
    int len = 0;
    for(;*text != '\0'; ++len, ++text);
    return len;
}

#define DEVICE_SYNC_CHECK_DP() \
{ \
    deviceError = cudaDeviceSynchronize(); \
}

__device__ cuHeapBuffer<CTuint> dp_nodes_ContentCount;
__device__ cuHeapBuffer<CTuint> dp_nodes_newContentCount;
__device__ cuHeapBuffer<CTuint> dp_nodes_ContentStartAdd;
__device__ cuHeapDoubleBuffer<BBox> dp_nodes_BBox;
__device__ cuHeapBuffer<CTbyte> dp_nodes_IsLeaf;
__device__ cuHeapBuffer<CTreal> dp_nodes_Split;
__device__ cuHeapBuffer<CTuint> dp_nodes_NodeIdToLeafIndex;
__device__ cuHeapBuffer<CTbyte> dp_nodes_SplitAxis;
__device__ cuHeapBuffer<CTuint> dp_nodes_LeftChild;
__device__ cuHeapBuffer<CTuint> dp_nodes_RightChild;

__device__ cuHeapBuffer<CTuint> dp_leafCountScanned;

__device__  cuHeapBuffer<CTbyte> dp_activeNodesIsLeaf;
__device__  cuHeapBuffer<CTuint> dp_activeNodes;
__device__  cuHeapBuffer<CTuint> dp_activeNodesThisLevel;

__device__  cuHeapBuffer<CTuint> dp_newActiveNodes;
__device__  cuHeapBuffer<CTuint> dp_newNodesContentCount;
__device__  cuHeapBuffer<CTuint> dp_newNodesContentStartAdd;

__device__ cuHeapBuffer<IndexedSAHSplit> dp_splits_IndexedSplit;
__device__ cuHeapBuffer<CTreal> dp_splits_Plane;
__device__ cuHeapBuffer<CTbyte> dp_splits_Axis;
__device__ cuHeapBuffer<CTuint> dp_splits_Above;
__device__ cuHeapBuffer<CTuint> dp_splits_Below;

__device__ cuHeapDoubleBuffer<CTuint> dp_eventNodeIndex;
__device__ cuHeapBuffer<CTbyte> dp_eventIsLeaf;
__device__ cuHeapBuffer<CTuint> dp_maskedInteriorContent;
__device__ cuHeapBuffer<CTuint> dp_scannedInteriorContent;

__device__ cuHeapBuffer<CTuint> dp_eventIsLeafScanned;
__device__ cuHeapBuffer<CTuint> dp_eventIsLeafScannedSums;

__device__ cuHeapBuffer<CTuint> dp_leafNodesContent;
__device__ cuHeapBuffer<CTuint> dp_leafContentCount;
__device__ cuHeapBuffer<CTuint> dp_leafContentStart;

struct dpClipMask
{
    cuHeapBuffer<CTbyte> mask0;
    cuHeapBuffer<CTbyte> mask1;
    cuHeapBuffer<CTbyte> mask2;

    cuHeapBuffer<CTuint> scannedMask0;
    cuHeapBuffer<CTuint> scannedMask1;
    cuHeapBuffer<CTuint> scannedMask2;

    cuHeapBuffer<CTuint> scannedSums0;
    cuHeapBuffer<CTuint> scannedSums1;
    cuHeapBuffer<CTuint> scannedSums2;

    cuHeapBuffer<CTreal> newSplits0;
    cuHeapBuffer<CTreal> newSplits1;
    cuHeapBuffer<CTreal> newSplits2;

    cuHeapBuffer<CTuint> index0;
    cuHeapBuffer<CTuint> index1;
    cuHeapBuffer<CTuint> index2;

    __device__ dpClipMask(void)
    {

    }

    __device__ ~dpClipMask(void)
    {

    }

    __device__ void init(void)
    {
        mask0.init();
        mask1.init();
        mask2.init();

        scannedMask0.init();
        scannedMask1.init();
        scannedMask2.init();

        scannedSums0.init();
        scannedSums1.init();
        scannedSums2.init();

        newSplits0.init();
        newSplits1.init();
        newSplits2.init();

        index0.init();
        index1.init();
        index2.init();
    }

    __device__ void resize(size_t newSize)
    {
        if(mask0.size() < newSize)
        {
           mask0.resize(newSize);
           mask1.resize(newSize);
           mask2.resize(newSize);

           scannedMask0.resize(newSize);
           scannedMask1.resize(newSize);
           scannedMask2.resize(newSize);

           scannedSums0.resize(newSize);
           scannedSums1.resize(newSize);
           scannedSums2.resize(newSize);

           newSplits0.resize(newSize);
           newSplits1.resize(newSize);
           newSplits2.resize(newSize);

           index0.resize(newSize);
           index1.resize(newSize);
           index2.resize(newSize);

           g_clipArray.mask[0].mask = mask0.GetPointer();
           g_clipArray.mask[1].mask = mask1.GetPointer();
           g_clipArray.mask[2].mask = mask2.GetPointer();

           g_clipArray.mask[0].newSplit = newSplits0.GetPointer();
           g_clipArray.mask[1].newSplit = newSplits1.GetPointer();
           g_clipArray.mask[2].newSplit = newSplits2.GetPointer();

           g_clipArray.mask[0].index = index0.GetPointer();
           g_clipArray.mask[1].index = index1.GetPointer();
           g_clipArray.mask[2].index = index2.GetPointer();

           g_clipArray.scanned[0] = scannedMask0.GetPointer();
           g_clipArray.scanned[1] = scannedMask1.GetPointer();
           g_clipArray.scanned[2] = scannedMask2.GetPointer();

           cms[0].mask = mask0.GetConstPointer();
           cms[1].mask = mask1.GetConstPointer();
           cms[2].mask = mask2.GetConstPointer();

           cms[0].newSplit = newSplits0.GetConstPointer();
           cms[1].newSplit = newSplits1.GetConstPointer();
           cms[2].newSplit = newSplits2.GetConstPointer();

           cms[0].index = index0.GetConstPointer();
           cms[1].index = index1.GetConstPointer();
           cms[2].index = index2.GetConstPointer();

           cms[0].scanned = scannedMask0.GetConstPointer();
           cms[1].scanned = scannedMask1.GetConstPointer();
           cms[2].scanned = scannedMask2.GetConstPointer();
        }
    }
};

struct dpEventLine
{
    cuHeapDoubleBuffer<IndexedEvent> indexedEvent;
    cuHeapDoubleBuffer<CTbyte> type;
    cuHeapDoubleBuffer<CTuint> primId;
    cuHeapDoubleBuffer<BBox> ranges;

    //nutty::Scanner typeStartScanner;
    cuHeapBuffer<CTuint> typeStartScanned;
    //nutty::Scanner eventScanner;
    cuHeapBuffer<CTuint> scannedEventTypeEndMask;
    cuHeapBuffer<CTuint> scannedEventTypeEndMaskSums;
    cuHeapBuffer<CTbyte> mask;
 
    CTbyte toggleIndex;

    __device__ dpEventLine(void)
    {

    }

    __device__ cuEventLine GetPtr(CTbyte index)
    {
        cuEventLine line;
        line.indexedEvent = indexedEvent[index].GetPointer();
        line.type = type[index].GetPointer();
        line.nodeIndex = dp_eventNodeIndex[index].GetPointer();
        line.primId = primId[index].GetPointer();
        line.ranges = ranges[index].GetPointer();
        line.mask = mask.GetPointer();
        line.scannedEventTypeEndMask = typeStartScanned.GetConstPointer();

        return line;
    }
    __device__ ~dpEventLine(void)
    {

    }

    __device__ void init(void)
    {
        indexedEvent.init();
        type.init();
        primId.init();
        ranges.init();
        typeStartScanned.init();
        scannedEventTypeEndMask.init();
        scannedEventTypeEndMaskSums.init();
        mask.init();
    }

    __device__ void resize(size_t newSize, CTbyte axis)
    {
        if(indexedEvent.size() < newSize)
        {
            indexedEvent.resize(newSize);
            type.resize(newSize);
            primId.resize(newSize);
            ranges.resize(newSize);
            typeStartScanned.resize(newSize);
            scannedEventTypeEndMask.resize(newSize);
            scannedEventTypeEndMaskSums.resize(newSize);
            mask.resize(newSize);

            g_eventTriples[0].lines[axis] = GetPtr(0);
            g_eventTriples[1].lines[axis] = GetPtr(1);
        }
    }
};

__device__ dpEventLine dpEventLines[3];
__device__ dpClipMask dpClipsMask;
__device__ CTbyte toggleIndex;

template <typename Operator, typename T>
__device__ void dpScanBinaryTriples(ConstTuple<3, T>& src, Tuple<3, CTuint>& scanned, Tuple<3, CTuint>& sums, CTuint N, Operator op)
{
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
}

__device__ void dpScanClipMaskTriples(CTuint eventCount)
{
    ConstTuple<3, CTbyte> ptr;
    ptr.ts[0] = dpClipsMask.mask0.GetConstPointer();
    ptr.ts[1] = dpClipsMask.mask1.GetConstPointer();
    ptr.ts[2] = dpClipsMask.mask2.GetConstPointer();

    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = dpClipsMask.scannedMask0.GetPointer();
    ptr1.ts[1] = dpClipsMask.scannedMask1.GetPointer();
    ptr1.ts[2] = dpClipsMask.scannedMask2.GetPointer();
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = dpClipsMask.scannedSums0.GetPointer();
    sums.ts[1] = dpClipsMask.scannedSums1.GetPointer();
    sums.ts[2] = dpClipsMask.scannedSums2.GetPointer();

    ClipMaskPrefixSumOP op;
    dpScanBinaryTriples(ptr, ptr1, sums, eventCount, op);
}

__device__ void dpScanEventTypesTriples(CTuint eventCount)
{
    CTuint eventBlock = 256U;
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    ConstTuple<3, CTbyte> ptr;
    ptr.ts[0] = dpEventLines[0].type[toggleIndex].GetConstPointer();
    ptr.ts[1] = dpEventLines[1].type[toggleIndex].GetConstPointer();
    ptr.ts[2] = dpEventLines[2].type[toggleIndex].GetConstPointer();

    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = dpEventLines[0].typeStartScanned.GetPointer();
    ptr1.ts[1] = dpEventLines[1].typeStartScanned.GetPointer();
    ptr1.ts[2] = dpEventLines[2].typeStartScanned.GetPointer();
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = dpEventLines[0].scannedEventTypeEndMaskSums.GetPointer();
    sums.ts[1] = dpEventLines[1].scannedEventTypeEndMaskSums.GetPointer();
    sums.ts[2] = dpEventLines[2].scannedEventTypeEndMaskSums.GetPointer();

    nutty::PrefixSumOp<CTbyte> op;

    dpScanBinaryTriples(ptr, ptr1, sums, eventCount, op); 
}

__device__ void toggle(void)
{
    if(threadIdx.x == 0)
    {
        toggleIndex ^= 1;
    }

    __syncthreads();
}

__device__ void GrowNodeMemory(CTuint newSize)
{
    if(threadIdx.x == 0)
    {
        newSize = newSize; //newSize ? dp_nodes_IsLeaf.size() * 4 : 1024;

        dp_nodes_IsLeaf.resize(newSize);
        dp_nodes_Split.resize(newSize);
        dp_nodes_NodeIdToLeafIndex.resize(newSize);
        dp_nodes_SplitAxis.resize(newSize);
        dp_nodes_LeftChild.resize(newSize);
        dp_nodes_RightChild.resize(newSize);
    
        g_nodes.isLeaf = dp_nodes_IsLeaf.GetPointer();
        g_nodes.splitAxis = dp_nodes_SplitAxis.GetPointer();
        g_nodes.split = dp_nodes_Split.GetPointer();
        g_nodes.contentStart = dp_nodes_ContentStartAdd.GetPointer();
        g_nodes.contentCount = dp_nodes_ContentCount.GetPointer();
        g_nodes.leftChild = dp_nodes_LeftChild.GetPointer();
        g_nodes.rightChild = dp_nodes_RightChild.GetPointer();
        g_nodes.nodeToLeafIndex = dp_nodes_NodeIdToLeafIndex.GetPointer();
    }
    __syncthreads();
}

__device__ void GrowPerLevelNodeMemory(CTuint newSize)
{
    if(threadIdx.x == 0)
    {
        dp_activeNodesIsLeaf.resize(newSize);
        dp_activeNodes.resize(newSize);
        dp_activeNodesThisLevel.resize(newSize);
        dp_newActiveNodes.resize(newSize);
        dp_newNodesContentCount.resize(newSize);
        dp_newNodesContentStartAdd.resize(newSize);
    
        dp_nodes_BBox.resize(newSize);
        dp_nodes_ContentStartAdd.resize(newSize);
        dp_nodes_ContentCount.resize(newSize);
        dp_nodes_newContentCount.resize(newSize);

        g_nodes.isLeaf = dp_nodes_IsLeaf.GetPointer();
        g_nodes.splitAxis = dp_nodes_SplitAxis.GetPointer();
        g_nodes.split = dp_nodes_Split.GetPointer();
        g_nodes.contentStart = dp_nodes_ContentStartAdd.GetPointer();
        g_nodes.contentCount = dp_nodes_ContentCount.GetPointer();
        g_nodes.leftChild = dp_nodes_LeftChild.GetPointer();
        g_nodes.rightChild = dp_nodes_RightChild.GetPointer();
        g_nodes.nodeToLeafIndex = dp_nodes_NodeIdToLeafIndex.GetPointer();
    }
    __syncthreads();
}

__device__ void GrowSplitMemory(CTuint eventCount)
{
    if(threadIdx.x == 0)
    {
        dp_splits_Above.resize(eventCount);
        dp_splits_Below.resize(eventCount);
        dp_splits_Axis.resize(eventCount);
        dp_splits_Plane.resize(eventCount);
        dp_splits_IndexedSplit.resize(eventCount);
        dp_eventNodeIndex.resize(eventCount);

        g_splits.above = dp_splits_Above.GetPointer();
        g_splits.below = dp_splits_Below.GetPointer();
        g_splits.axis = dp_splits_Axis.GetPointer();
        g_splits.indexedSplit = dp_splits_IndexedSplit.GetPointer();
        g_splits.v = dp_splits_Plane.GetPointer();

        dp_eventNodeIndex.resize(eventCount);
        dp_eventIsLeaf.resize(eventCount);

        g_splitsConst.above = dp_splits_Above.GetConstPointer();
        g_splitsConst.below = dp_splits_Below.GetConstPointer();
        g_splitsConst.axis = dp_splits_Axis.GetConstPointer();
        g_splitsConst.indexedSplit = dp_splits_IndexedSplit.GetConstPointer();
        g_splitsConst.v = dp_splits_Plane.GetConstPointer();
    }
    __syncthreads();
}

__global__ void dpInit(CTuint primitiveCount, const BBox* __restrict sceneBBox, const BBox* __restrict primBBox, CTbyte initMemory)
{
    if(threadIdx.x == 0)
    {
        if(initMemory)
        {
            GrowNodeMemory(2048);
            GrowPerLevelNodeMemory(2048);
            GrowSplitMemory(4 * primitiveCount);

            dpEventLines[0].init();
            dpEventLines[1].init();
            dpEventLines[2].init();

            dp_maskedInteriorContent.resize(2048);
            dp_scannedInteriorContent.resize(2048);

            dpEventLines[0].resize(4 * primitiveCount, 0);
            dpEventLines[1].resize(4 * primitiveCount, 1);
            dpEventLines[2].resize(4 * primitiveCount, 2);

            dp_eventIsLeafScanned.resize(4 * primitiveCount);
            dp_eventIsLeafScannedSums.resize(4 * primitiveCount/256 + 256);
            dp_leafNodesContent.resize(4 * primitiveCount/2);
            dp_leafContentStart.resize(2048);
            dp_leafContentCount.resize(2048);

            dpClipsMask.resize(4 * primitiveCount);

            if(mem_alloc_error)
            {
                return;
            }
        }

        toggleIndex = 0;

        CTuint elementBlock = 256U;
        CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);

        dp_nodes_ContentStartAdd[0] = 0;
        dp_nodes_ContentCount[0] = primitiveCount;

        createEventsAndInit3<1, 0><<<elementGrid, elementBlock>>>(
            primBBox, 
            sceneBBox,

            dp_activeNodes.GetPointer(),
            dp_nodes_NodeIdToLeafIndex.GetPointer(),
            dp_nodes_IsLeaf.GetPointer(),
            dp_nodes_ContentCount.GetPointer(),
            dp_nodes_BBox[0].GetPointer(),

            primitiveCount);

        for(CTbyte i = 0; i < 3; ++i)
        {
            nutty::Sort(
                nutty::DevicePtr_Cast<IndexedEvent>(dpEventLines[i].indexedEvent[0]._pMemPtr), 
                nutty::DevicePtr_Cast<IndexedEvent>(dpEventLines[i].indexedEvent[0]._pMemPtr + 2 * primitiveCount), 
                EventSort());
        }

        reorderEvent3<<<2 * elementGrid, elementBlock>>>(2 * primitiveCount);

        toggleIndex = 1;
        
        mem_alloc_error = 0;
        deviceError = 0;
    }
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

__device__ void dpCcomputeSAH_Splits(CTuint nodeCount, CTuint eventCount, CTuint d = 0)
{
    dpScanEventTypesTriples(eventCount);

    const CTuint elemsPerThread = 1;
    CTuint sahBlock = 256U;
    CTuint sahGrid = nutty::cuda::GetCudaGrid(eventCount, sahBlock);

    computeSAHSplits3<1, elemsPerThread><<<sahGrid, sahBlock>>>(
        dp_nodes_ContentCount.GetConstPointer(),
        dp_nodes_ContentStartAdd.GetConstPointer(),
        dp_nodes_BBox[0].GetConstPointer(),
        eventCount,
        toggleIndex);

    const CTuint blockSize = 512U;
    CTuint N = nodeCount * blockSize;
//     CTuint reduceGrid = nutty::cuda::GetCudaGrid(N, blockSize);
    
    //deviceError = nodeCount;

    segReduce<blockSize><<<nodeCount, blockSize>>>(dp_splits_IndexedSplit.GetPointer(), N, eventCount);
}

template<CTuint BLOCK_SIZE>
__global__ void dpCreateKDTree(CTuint primitiveCount, CTuint maxDepth)
{
    CTuint nodesCount = 1;
    CTuint eventCount = 2 * primitiveCount;

    CTuint interiorNodesCountOnThisLevel = 1;
    CTuint currentInteriorNodesCount = 1;
    CTuint nodeOffset = 0;
    CTuint leafContentOffset = 0;
    CTuint currentLeafCount = 0;
    CTuint childNodeOffset = 1;
    CTuint leafPrimitiveCount = 0;

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

            dpEventLines[0].resize(2 * count, 0);
            dpEventLines[1].resize(2 * count, 1);
            dpEventLines[2].resize(2 * count, 2);
            
            dpClipsMask.resize(2 * count);

            createClipMask<<<grid, block>>>(
                dp_nodes_ContentStartAdd.GetPointer(), 
                dp_nodes_ContentCount.GetPointer(),
                count,
                toggleIndex);

            dpScanClipMaskTriples(2 * count);

            CTuint _block = 256U;
            CTuint _grid = nutty::cuda::GetCudaGrid(2 * count, block);

            compactEventLineV2<<<_grid, _block>>>(
                2 * count, toggleIndex);
            
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
                CTuint edgesBeforeMe = 2 * dp_nodes_ContentStartAdd[id];
                CTuint eventOffset = 2 * dp_nodes_ContentCount[id];
              
                IndexedSAHSplit split = g_splitsConst.indexedSplit[edgesBeforeMe];

                CTbyte axis = g_splitsConst.axis[split.index];
                CTreal s = g_splitsConst.v[split.index];

                CTuint below = g_splitsConst.below[split.index];
                CTuint above = g_splitsConst.above[split.index];

                CTuint nodeId = nodeOffset + dp_activeNodes[id];

                g_nodes.split[nodeId] = s;
                g_nodes.splitAxis[nodeId] = axis;

                CTuint dst = id;
     
                dp_nodes_newContentCount[2 * dst + 0] = below;
                dp_nodes_newContentCount[2 * dst + 1] = above;
      
                CTuint leftChildIndex = childNodeOffset + 2 * id + 0;
                CTuint rightChildIndex = childNodeOffset + 2 * id + 1;

                g_nodes.leftChild[nodeId] = leftChildIndex;
                g_nodes.rightChild[nodeId] = rightChildIndex;

//                 if((makeLeaves || (below <= MAX_ELEMENTS_PER_LEAF || above <= MAX_ELEMENTS_PER_LEAF)))
//                 {
//                     gotLeaves[0] = 1;
//                 }

                g_nodes.isLeaf[childNodeOffset + 2 * id + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                g_nodes.isLeaf[childNodeOffset + 2 * id + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;

                dp_activeNodesIsLeaf[nodesCount + 2 * id + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                dp_activeNodesIsLeaf[nodesCount + 2 * id + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
                
                dp_newActiveNodes[2 * id + 0] = 2 * id + 0;
                dp_newActiveNodes[2 * id + 1] = 2 * id + 1;

                BBox l;
                BBox r;

                splitAABB(&dp_nodes_BBox[0][id], s, axis, &l, &r);

                if(below > MAX_ELEMENTS_PER_LEAF)
                {
                    dp_nodes_BBox[1][2 * dst + 0] = l;
                }

                if(above > MAX_ELEMENTS_PER_LEAF)
                {
                    dp_nodes_BBox[1][2 * dst + 1] = r;
                }
            }
        }

        __syncthreads();
   
        for(int offset = 0; offset < childCount; offset += blockDim.x)
        {
            CTuint id = offset + threadIdx.x; 
            if(id < childCount)
            {
                dp_nodes_ContentCount[id] = dp_nodes_newContentCount[id];
                dp_activeNodes[id] = dp_newActiveNodes[id];
            }
        }

        __syncthreads();

        eventCount = dpClipsMask.scannedMask2[2 * count - 1] + isSet(dpClipsMask.mask2[2 * count - 1]);

        if(threadIdx.x == 0)
        {
            CTuint eventBlock = 256U;
            CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
            
            setEventsBelongToLeafAndSetNodeIndex<<<eventGrid, eventBlock>>>(
                dp_activeNodesIsLeaf.GetPointer() + nodesCount,
                dp_eventIsLeaf.GetPointer(),
                dp_nodes_NodeIdToLeafIndex.GetPointer() + childNodeOffset,
                eventCount,
                2 * nodesCount,
                toggleIndex);

            cudaDeviceSynchronize();
        }

        //__syncthreads();
        
        completeBlockScan<BLOCK_SIZE>(
            dp_nodes_ContentCount.GetConstPointer(), 
            dp_nodes_ContentStartAdd.GetPointer(), 
            shrdMem, nutty::PrefixSumOp<CTuint>(), 
            childCount);

        dp_leafCountScanned.resize(childCount);

        __syncthreads();
        
        completeBlockScan<BLOCK_SIZE>(
                    dp_activeNodesIsLeaf.GetConstPointer() + nodesCount, 
                    dp_leafCountScanned.GetPointer(), 
                    shrdMem, TypeOp<CTbyte>(), 
                    childCount);

        //__syncthreads();

        CTuint leafCount = dp_leafCountScanned[childCount - 1] + (CTuint)dp_activeNodesIsLeaf[nodesCount + childCount - 1];
  
        if(leafCount)
        {
            dp_maskedInteriorContent.resize(childCount);
            dp_scannedInteriorContent.resize(childCount);

            __syncthreads();

            for(int offset = 0; offset < childCount; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < childCount)
                {
                    CTbyte mask = dp_activeNodesIsLeaf[nodesCount + id];
                    dp_maskedInteriorContent[id] = /*(mask < 2) */ (1 ^ mask) * dp_nodes_ContentCount[id];
                }
            }

            //__syncthreads();

            completeBlockScan<BLOCK_SIZE>(
                        dp_maskedInteriorContent.GetConstPointer(), 
                        dp_scannedInteriorContent.GetPointer(), 
                        shrdMem, nutty::PrefixSumOp<CTuint>(), 
                        childCount);

            //__syncthreads();

            const CTuint eventBlock = EVENT_GROUP_SIZE;
            CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
    
            dp_eventIsLeafScanned.resize(eventCount);
            dp_eventIsLeafScannedSums.resize(eventCount/256 + 256);

            __syncthreads();
     
            if(threadIdx.x == 0)
            {

                binaryGroupScan<256><<<eventGrid, eventBlock>>>(
                    dp_eventIsLeaf.GetConstPointer(), dp_eventIsLeafScanned.GetPointer(), dp_eventIsLeafScannedSums.GetPointer(), TypeOp<CTbyte>(), eventCount);

                CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);
                if(sumsCount > 1)
                {
                    nutty::PrefixSumOp<CTuint> _op;
                    completeScan<256><<<1, 256>>>(dp_eventIsLeafScannedSums.GetConstPointer(), dp_eventIsLeafScannedSums.GetPointer(), _op, sumsCount);
                
                    spreadScannedSumsSingle<<<eventGrid-1, eventBlock>>>(
                            dp_eventIsLeafScanned.GetPointer(), dp_eventIsLeafScannedSums.GetConstPointer(), eventCount);
                }
                cudaDeviceSynchronize();
            }
            
            //CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);

            __syncthreads();
 
//             if(sumsCount > 1)
//             {
//                 nutty::PrefixSumOp<CTuint> _op;
//                 completeBlockScan<BLOCK_SIZE>(dp_eventIsLeafScannedSums.GetConstPointer(), dp_eventIsLeafScannedSums.GetPointer(), shrdMem, _op, sumsCount);
// 
//                 if(threadIdx.x == 0)
//                 {
//                     spreadScannedSumsSingle<<<eventGrid-1, eventBlock>>>(
//                             dp_eventIsLeafScanned.GetPointer(), dp_eventIsLeafScannedSums.GetConstPointer(), eventCount);
//                     cudaDeviceSynchronize();
//                 }
//             }
            
            CTuint nodeBlock = NODES_GROUP_SIZE;
            CTuint nodeGrid = nutty::cuda::GetCudaGrid(childCount, nodeBlock);

            dp_leafNodesContent.resize(leafContentOffset + eventCount/2);
            dp_leafContentStart.resize(currentLeafCount + leafCount);
            dp_leafContentCount.resize(currentLeafCount + leafCount);
            
            __syncthreads();

            if(threadIdx.x == 0)
            {
                compactMakeLeavesData<0><<<eventGrid, eventBlock>>>(
                    dp_activeNodesIsLeaf.GetConstPointer() + nodesCount,
     
                    dp_nodes_ContentStartAdd.GetConstPointer(),
         
                    dp_eventIsLeafScanned.GetConstPointer(),
            
                    dp_nodes_ContentCount.GetPointer(),
                    dp_eventIsLeaf.GetConstPointer(),

                    dp_leafCountScanned.GetConstPointer(),

                    dp_activeNodes.GetConstPointer(),
                    dp_leafCountScanned.GetConstPointer(),
                    dp_scannedInteriorContent.GetConstPointer(),
                    dp_nodes_BBox[1].GetConstPointer(),

                    dp_leafNodesContent.GetPointer(),
                    dp_nodes_NodeIdToLeafIndex.GetPointer(),
                    dp_newNodesContentCount.GetPointer(),
                    dp_newNodesContentStartAdd.GetPointer(),
                    dp_leafContentStart.GetPointer(),
                    dp_leafContentCount.GetPointer(),
                    dp_newActiveNodes.GetPointer(),
                    dp_nodes_BBox[0].GetPointer(),
         
                    childNodeOffset,
                    leafContentOffset,
                    currentLeafCount,
                    childCount,
                    toggleIndex,
                    eventCount);

                cudaDeviceSynchronize();
            }

            for(int offset = 0; offset < childCount; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < childCount)
                {
                    CTbyte leafMask = dp_activeNodesIsLeaf[nodesCount + id];
                    CTuint dst = dp_leafCountScanned[id];
                    if(leafMask)
                    {
                        dp_leafContentStart[currentLeafCount + dst] = leafContentOffset + (dp_nodes_ContentStartAdd[id] - dp_scannedInteriorContent[id]);
                        dp_leafContentCount[currentLeafCount + dst] = dp_nodes_ContentCount[id];
                        dp_nodes_NodeIdToLeafIndex[dp_activeNodes[id] + childNodeOffset] = currentLeafCount + dp_leafCountScanned[id];
                    }
                    else
                    {
                        dst = id - dst;
                        dp_newActiveNodes[dst] = dp_activeNodes[id];
                        dp_newNodesContentCount[dst] = dp_nodes_ContentCount[id];
                        dp_newNodesContentStartAdd[dst] = dp_scannedInteriorContent[id];
                        dp_nodes_BBox[0][dst] = dp_nodes_BBox[1][id];
                    }
                }
            }

            __syncthreads();

            CTuint copyDistance = childCount - leafCount;
            CTuint interiorPrimCount = dp_scannedInteriorContent[childCount-1] +
                dp_nodes_ContentCount[childCount-1] * (dp_activeNodesIsLeaf[childCount + nodesCount - 1] == 0);

            leafPrimitiveCount = eventCount/2 - interiorPrimCount;

            eventCount = 2 * interiorPrimCount;

            //__syncthreads();

            for(int offset = 0; offset < copyDistance; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < copyDistance)
                {
                    dp_nodes_ContentCount[id] = dp_newNodesContentCount[id];
                    dp_nodes_ContentStartAdd[id] = dp_newNodesContentStartAdd[id];
                    dp_activeNodes[id] = dp_newActiveNodes[id];
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
            for(int offset = 0; offset < interiorNodesCountOnThisLevel; offset += blockDim.x)
            {
                CTuint id = offset + threadIdx.x;
                if(id < interiorNodesCountOnThisLevel)
                {
                    dp_nodes_BBox[0][id] = dp_nodes_BBox[1][id];
                }
            }
        }

        if(d < maxDepth-1) //are we not done?
        {
            //check if we need more memory
            if(eventCount > dp_splits_Above.size())
            {
                GrowSplitMemory(4 * eventCount);
            }

            if(dp_activeNodes.size() < interiorNodesCountOnThisLevel + 2 * interiorNodesCountOnThisLevel)
            {
                GrowPerLevelNodeMemory(4 * 2 * interiorNodesCountOnThisLevel);
            }

            if(dp_nodes_IsLeaf.size() < (childNodeOffset + 2 * interiorNodesCountOnThisLevel))
            {
                GrowNodeMemory(4 * dp_nodes_IsLeaf.size());
            }
        }
                                         
        __syncthreads();
    }
}

__global__ void dpFreeKDTreeMemory(void)
{

}

template<typename T>
__global__ void cpyHeapMem(T** dst, CT_LINEAR_MEMORY_TYPE type)
{
    switch(type)
    {
        case eCT_LEAF_NODE_PRIM_IDS :
            {
                dst[0] = (T*)dp_leafNodesContent._pMemPtr;
            } break;
        case eCT_LEAF_NODE_PRIM_START_INDEX :
            {
                dst[0] = (T*)dp_leafContentStart._pMemPtr;
            } break;
        case eCT_LEAF_NODE_PRIM_COUNT :
            {
                dst[0] = (T*)dp_leafContentCount._pMemPtr;
            } break;
        case eCT_NODE_SPLITS :
            {
                dst[0] = (T*)dp_nodes_Split._pMemPtr;
            } break;
        case eCT_NODE_SPLIT_AXIS :
            {
                dst[0] = (T*)dp_nodes_SplitAxis._pMemPtr;
            } break;
        case eCT_NODE_RIGHT_CHILD :
            {
                dst[0] = (T*)dp_nodes_RightChild._pMemPtr;
            } break;
        case eCT_NODE_LEFT_CHILD :
            {
                dst[0] = (T*)dp_nodes_LeftChild._pMemPtr;
            } break;
        case eCT_NODE_IS_LEAF :
            {
                dst[0] = (T*)dp_nodes_IsLeaf._pMemPtr;
            } break;
        case eCT_NODE_TO_LEAF_INDEX :
            {
                dst[0] = (T*)dp_nodes_NodeIdToLeafIndex._pMemPtr;
            } break;
    }
}

const void* cudpKDTree::GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const
{
    switch(type)
    {
    case eCT_LEAF_NODE_PRIM_IDS :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTuint*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_LEAF_NODE_PRIM_IDS);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_LEAF_NODE_PRIM_START_INDEX :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTuint*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_LEAF_NODE_PRIM_START_INDEX);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_LEAF_NODE_PRIM_COUNT :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTuint*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_LEAF_NODE_PRIM_COUNT);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_NODE_PRIM_IDS :
        {
            *byteCount = 0;
            return NULL;
        }
    case eCT_PRIMITVES :
        {
            return GetRawPrimitives(byteCount);
        }
    case eCT_NODE_SPLITS :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTreal*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_NODE_SPLITS);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_NODE_SPLIT_AXIS :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTbyte*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_NODE_SPLIT_AXIS);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_NODE_RIGHT_CHILD :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTuint*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_NODE_RIGHT_CHILD);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_NODE_LEFT_CHILD :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTuint*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_NODE_LEFT_CHILD);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_NODE_IS_LEAF :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTbyte*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_NODE_IS_LEAF);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    case eCT_NODE_PRIM_COUNT :
        {
            *byteCount = 0;
            return NULL;
        }
    case eCT_NODE_PRIM_START_INDEX :
        {
            *byteCount = 0;
            return NULL;
        }
    case eCT_NODE_TO_LEAF_INDEX :
        {
            *byteCount = 0;
            nutty::DeviceBuffer<CTuint*> dptr(1);
            cpyHeapMem<<<1,1>>>(dptr.Begin()(), eCT_NODE_TO_LEAF_INDEX);
            DEVICE_SYNC_CHECK();
            return dptr[0];
        }
    default : *byteCount = 0; return NULL;
    }
}


CT_RESULT cudpKDTree::Update(void)
{
    if(!m_initialized)
    {
        size_t limit;
        cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));
    }
    
    CTuint primitiveCount = m_currentTransformedVertices.Size() / 3;

    //static bool staticc = true;

    m_primAABBs.Resize(primitiveCount);

    cudaCreateTriangleAABBs(m_currentTransformedVertices.GetPointer(), m_primAABBs.GetPointer(), primitiveCount, m_pStream);

   // if(staticc)
    {
        DEVICE_SYNC_CHECK();

        static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
        static float3 min3f = -max3f;

        BBox bboxN;
        bboxN.m_min = max3f; 
        bboxN.m_max = min3f;
        m_sceneBBox.Resize(m_primAABBs.Size()/2);
        nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), ReduceBBox(), bboxN, m_pStream);
      //  staticc = false;
    }

    dpInit<<<1, 1, 0, m_pStream>>>(primitiveCount, m_sceneBBox.GetPointer(), m_primAABBs.GetPointer(), !m_initialized);

    if(!m_initialized)
    {
        m_initialized = true;
        m_depth = (byte)min(64, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));
    }

    CTuint d_error = NULL;
    CTuint h_error = NULL;

    cudaMemcpyFromSymbol(&h_error, mem_alloc_error, sizeof(CTuint));

    if(h_error)
    {
        __ct_printf("%s\n", cudaGetErrorString((cudaError_t)h_error));
        exit(0);
    }

    DEVICE_SYNC_CHECK();

    const CTuint BLOCK_SIZE = 256;
    dpCreateKDTree<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, m_pStream>>>(primitiveCount, m_depth);

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyFromSymbol(&h_error, deviceError, sizeof(uint)));

    __ct_printf("%d\n", h_error);

//     CTreal h_msgBuffer[256];
//     cudaMemcpyFromSymbol(&h_msgBuffer, msgBuffer, sizeof(h_msgBuffer));
// 
//     for(int i = 0; i < sizeof(h_msgBuffer)/4; ++i)
//     {
//         __ct_printf("%.0f ", h_msgBuffer[i]);
//     }
// 
//     __ct_printf("\n");

    DEVICE_SYNC_CHECK();

//     char msg[256];
//     char* d_msg;
//     cudaGetSymbolAddress((void**)&d_msg, errorLog);
//     cudaMemcpy(msg, d_msg, 256, cudaMemcpyDeviceToHost);

//     cudaError_t cerror;
//     cudaError_t* d_cerror;
//     cudaGetSymbolAddress((void**)&d_cerror, deviceError);
//     cudaMemcpy(&cerror, d_cerror, 1, cudaMemcpyDeviceToHost);
// 
//     CUDA_RT_SAFE_CALLING_NO_SYNC(cerror);

    return CT_SUCCESS;
}

void cudpKDTree::GrowPerLevelNodeMemory(CTuint newSize)
{

}

void cudpKDTree::GrowNodeMemory(void)
{

}

void cudpKDTree::GrowSplitMemory(CTuint eventCount)
{

}