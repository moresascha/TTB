#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"
#include <Reduce.h>
#include <device_functions.h>
#include <sm_35_intrinsics.h>

#undef DYNAMIC_PARALLELISM

#ifdef _DEBUG
#undef DYNAMIC_PARALLELISM
#endif

template<CTuint width>
__device__ int blockScan(CTuint* sums, int value)
{
    int warp_id = threadIdx.x / warpSize;
    int lane_id = laneid();
#pragma unroll
    for (CTuint i=1; i<width; i*=2)
    {
        int n = __shfl_up(value, i, width);

        if (lane_id >= i) value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize-1)
    {
        sums[warp_id] = value;
    }

    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0)
    {
        int warp_sum = sums[lane_id];

        for (int i=1; i<width; i*=2)
        {
            int n = __shfl_up(warp_sum, i, width);

            if (laneid() >= i) warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    int blockSum = 0;

    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    return value;
}

template <CTbyte hasLeaves>
__global__ void setActiveNodesMask(
    CTuint* ids,
    const CTnodeIsLeaf_t* __restrict isLeaf,
    const CTuint* __restrict prefixSum,
    CTuint nodeOffset,
    CTuint N
    )
{
    RETURN_IF_OOB(N);

    if(hasLeaves)
    {
        if(!isLeaf[nodeOffset + id])
        {
            ids[prefixSum[id]] = id;
        }
    }
    else
    {
        ids[id] = id;
    }
}

void cudaSetActiveNodesMask(CTuint* ids,
                            const CTnodeIsLeaf_t* __restrict isLeaf,
                            const CTuint* __restrict prefixSum,
                            CTuint nodeOffset,
                            CTuint N, CTuint grid, CTuint block, cudaStream_t stream)
{
    setActiveNodesMask<1><<<grid, block, 0, stream>>>(
        ids, 
        isLeaf, 
        prefixSum,
        0, 
        N);
}

template <typename T>
__global__ void makeInverseScan(const T* __restrict scanVals, T* dst, CTuint N)
{
    RETURN_IF_OOB(N);

    dst[id] = id - scanVals[id];
}


struct tmp0
{
    const CTuint* __restrict v[3];
};

struct tmp1
{
    CTuint* dst[3];
};

#define DP_VERSION

#ifdef DP_VERSION
#define CUDA_CONSTANT_ __device__
#else
#define CUDA_CONSTANT_ __constant__
#endif


/*__constant__ cuEventLineTriple g_eventTripleDst;
__constant__ cuEventLineTriple g_eventTripleSrc; */
CUDA_CONSTANT_ cuEventLineTriple g_eventTriples[2]; //src-dst

CUDA_CONSTANT_ cuConstEventLineTriple g_eventSrcTriples;
CUDA_CONSTANT_ cuConstEventLineTriple g_eventDstTriples;

CUDA_CONSTANT_ Split g_splits;
CUDA_CONSTANT_ SplitConst g_splitsConst;
CUDA_CONSTANT_ Node g_nodes;

CUDA_CONSTANT_ cuClipMaskArray g_clipArray;
struct Sums
{
    const CTuint* __restrict prefixSum[3];
};

CUDA_CONSTANT_ cuConstClipMask cms[3];

/*__constant__ CTbyte g_eventSrcIndex;
__constant__ CTbyte g_eventDstIndex;*/

#define EVENT_TRIPLE_HEADER_SRC const cuEventLineTriple& __restrict eventsSrc = g_eventTriples[srcIndex]
#define EVENT_TRIPLE_HEADER_DST cuEventLineTriple& eventsDst = g_eventTriples[(srcIndex+1)%2]
// #define eventsSrc g_eventTriples[srcIndex]
// #define eventsDst g_eventTriples[(srcIndex+1)%2]

__global__ void makeInverseScanN(tmp0 scanVals, tmp1 dst, CTuint N)
{
    RETURN_IF_OOB(N);

#pragma unroll
    for(int k = 0; k < 3; ++k)
    {
        dst.dst[k][id] = id - scanVals.v[k][id];
    }
}

__global__ void makeOthers(const CTuint* nodesContentScanned, const CTuint* __restrict interiorScannedContent, CTuint* leafContentScanned, 
                           
                           const CTuint* __restrict leafCountScanned, CTuint* interiorCountScanned,

                           CTuint N)
{
    RETURN_IF_OOB(N);

    interiorCountScanned[id] = id - leafCountScanned[id];
    leafContentScanned[id] = nodesContentScanned[id] - interiorScannedContent[id];
}

struct RawEventData
{
    float* rawEvents[3];
    unsigned int* rawEventkeys[3];
};

__global__ void reorderEvent3(
    RawEventData eventData,
    CTuint N)
{
    RETURN_IF_OOB(N);

    #pragma unroll
    for(CTaxis_t a = 0; a < 3; ++a)
    {
        CTuint srcIndex = eventData.rawEventkeys[a][id];//g_eventTriples[0].lines[a].indexedEvent[id].index;
        IndexedEvent event;
        //event.index = srcIndex;
        event.v = eventData.rawEvents[a][id];

        g_eventTriples[1].lines[a].indexedEvent[id] = event;//g_eventTriples[0].lines[a].indexedEvent[id];

        g_eventTriples[1].lines[a].type[id] = g_eventTriples[0].lines[a].type[srcIndex];

        g_eventTriples[1].lines[a].nodeIndex[id] = 0;//g_eventTriples[0].lines[a].nodeIndex[srcIndex];

        g_eventTriples[1].lines[a].primId[id] = g_eventTriples[0].lines[a].primId[srcIndex];

        g_eventTriples[1].lines[a].ranges[id] = g_eventTriples[0].lines[a].ranges[srcIndex];
    }
}

void cudaReorderEvent3(CTuint N, CTuint grid, CTuint block, float* rawEvents[3], unsigned int* rawEventkeys[3], cudaStream_t stream)
{
    RawEventData eventData;
    eventData.rawEvents[0] = rawEvents[0];
    eventData.rawEvents[1] = rawEvents[1];
    eventData.rawEvents[2] = rawEvents[2];

    eventData.rawEventkeys[0] = rawEventkeys[0];
    eventData.rawEventkeys[1] = rawEventkeys[1];
    eventData.rawEventkeys[2] = rawEventkeys[2];

    reorderEvent3<<<grid, block, 0, stream>>>(eventData, N);
}

template<CTbyte useFOR, CTbyte axis>
__global__ void createEventsAndInit3(
    const BBox* __restrict primAxisAlignedBB,
    const BBox* __restrict sceneBBox,

    CTuint* activeNodes,
    CTuint* nodeToLeafIndex,
    CTnodeIsLeaf_t* nodeIsLeaf,
    CTuint* nodesContentCount,
    BBox* nodesBBox,

    RawEventData eventData,

    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint primIndex = id;
    BBox aabb = primAxisAlignedBB[primIndex];

    CTuint start_index = 2 * id + 0;
    CTuint end_index = 2 * id + 1;

    if(useFOR)
    {
        #pragma unroll
        for(CTaxis_t a = 0; a < 3; ++a)
        {
            g_eventTriples[0].lines[a].type[start_index] = EVENT_START;
            g_eventTriples[0].lines[a].type[end_index] = EDGE_END;

            g_eventTriples[0].lines[a].nodeIndex[start_index] = 0;
            g_eventTriples[0].lines[a].nodeIndex[end_index] = 0;

            eventData.rawEventkeys[a][start_index] = start_index;
            eventData.rawEventkeys[a][end_index] = end_index;

//             g_eventTriples[0].lines[a].indexedEvent[start_index].index = start_index;
//             g_eventTriples[0].lines[a].indexedEvent[end_index].index = end_index;

            g_eventTriples[0].lines[a].primId[start_index] = primIndex;
            g_eventTriples[0].lines[a].primId[end_index] = primIndex;

            eventData.rawEvents[a][start_index] = getAxis(aabb.m_min, a);
            eventData.rawEvents[a][end_index] = getAxis(aabb.m_max, a);
            g_eventTriples[0].lines[a].indexedEvent[start_index].v = getAxis(aabb.m_min, a);
            g_eventTriples[0].lines[a].indexedEvent[end_index].v = getAxis(aabb.m_max, a);

            g_eventTriples[0].lines[a].ranges[start_index] = aabb;
            g_eventTriples[0].lines[a].ranges[end_index] = aabb;
        }
    }

    if(threadIdx.x == 0)
    {
        g_nodes.contentStart[0] = 0;
        activeNodes[0] = 0;
        nodeToLeafIndex[0] = 0;
        nodesBBox[0] = sceneBBox[0];
        nodesContentCount[0] = N;
        nodeIsLeaf[0] = 0;
    }
}

void cudaCreateEventsAndInit3(const BBox* __restrict primAxisAlignedBB,
                             const BBox* __restrict sceneBBox,

                             CTuint* activeNodes,
                             CTuint* nodeToLeafIndex,
                             CTnodeIsLeaf_t* nodeIsLeaf,
                             CTuint* nodesContentCount,
                             BBox* nodesBBox,
                             float* rawEvents[3], unsigned int* rawEventkeys[3],
                             CTuint N, 
                             CTuint grid,
                             CTuint block,
                             cudaStream_t stream)
{
    RawEventData eventData;
    eventData.rawEvents[0] = rawEvents[0];
    eventData.rawEvents[1] = rawEvents[1];
    eventData.rawEvents[2] = rawEvents[2];

    eventData.rawEventkeys[0] = rawEventkeys[0];
    eventData.rawEventkeys[1] = rawEventkeys[1];
    eventData.rawEventkeys[2] = rawEventkeys[2];

    createEventsAndInit3<1, 0><<<grid, block, 0, stream>>>(
        primAxisAlignedBB, 
        sceneBBox, activeNodes, 
        nodeToLeafIndex, 
        nodeIsLeaf, 
        nodesContentCount, 
        nodesBBox, 
        eventData,
        N);
}

__global__ void computeSAHSplits3Old(
    const CTuint* __restrict nodesContentCount,
    const CTuint* __restrict nodesContentStart,
    const BBox* __restrict nodesBBoxes,
    CTuint N,
    CTbyte srcIndex) 
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex;
    BBox bbox;
    CTaxis_t axis;
    CTreal r = -1;

    EVENT_TRIPLE_HEADER_SRC;

#pragma unroll
    for(CTaxis_t i = 0; i < 3; ++i)
    {
        CTuint ni = eventsSrc.lines[i].nodeIndex[id];
        BBox _bbox = nodesBBoxes[ni];
        CTreal d = getAxis(_bbox.m_max, i) - getAxis(_bbox.m_min, i);
        if(d > r)
        {
            r = d;
            nodeIndex = ni;
            bbox = _bbox;
            axis = i;
        }
    }

    CTeventType_t type = eventsSrc.lines[axis].type[id];

    CTreal split = eventsSrc.lines[axis].indexedEvent[id].v;

    CTuint prefixPrims = nodesContentStart[nodeIndex]; //events.lines[axis].prefixSum[id];

    CTuint primCount = nodesContentCount[nodeIndex];

//     CTuint startScan = eventsSrc.lines[axis].scannedEventTypeStartMask[id];
//     CTuint endScan = id - startScan;
//     CTuint above = primCount - endScan + prefixPrims - type;
//     CTuint below = startScan - prefixPrims;

    CTuint startScan = eventsSrc.lines[axis].scannedEventTypeStartMask[id];
    CTuint endScan = id - startScan;
    CTuint above = primCount - (id - startScan - prefixPrims) - type;
    CTuint below = startScan - prefixPrims;

//     CTuint startScan = eventsSrc.lines[axis].scannedEventTypeStartMask[id];
//     CTuint above = primCount - /*events.lines[axis].scannedEventTypeEndMask[id]*/ (id - startScan) + prefixPrims - type;
//     CTuint below = startScan - prefixPrims;

    g_splits.above[id] = above;
    g_splits.below[id] = below;
    g_splits.indexedSplit[id].index = id;

    g_splits.indexedSplit[id].sah = getSAH(bbox, axis, split, below, above);
    g_splits.axis[id] = axis;
    g_splits.v[id] = split;
}

struct SAHInfoCache
{
    CTuint nodeIndex;
    BBox bbox;
    CTuint prefixPrims;
    CTuint primCount;
};

template<bool LONGEST_AXIS, CTuint elemsPerThread>
__global__ void computeSAHSplits3(
    const CTuint* __restrict nodesContentCount,
    const CTuint* __restrict nodesContentStart,
    const BBox* __restrict nodesBBoxes,
    CTuint N,
    CTbyte srcIndex) 
{
    RETURN_IF_OOB(N);

    EVENT_TRIPLE_HEADER_SRC;
    CTuint offset = 0;

    SAHInfoCache cache[elemsPerThread];

#pragma unroll
    for(CTuint i = 0; i < elemsPerThread; ++i)
    {
        CTuint idx = id + offset;

        if(idx >= N)
        {
            cache[i].nodeIndex = -1U;
            break;
        }

        offset += gridDim.x * blockDim.x;

        cache[i].nodeIndex = eventsSrc.lines[0].nodeIndex[idx];
    }

    offset = 0;
#pragma unroll
    for(CTuint i = 0; i < elemsPerThread; ++i)
    {
        if(cache[i].nodeIndex != -1U)
        {
            cache[i].bbox = nodesBBoxes[cache[i].nodeIndex];

            cache[i].prefixPrims = nodesContentStart[cache[i].nodeIndex];

            cache[i].primCount = nodesContentCount[cache[i].nodeIndex];
        }
    }

#pragma unroll
    for(CTuint i = 0; i < elemsPerThread; ++i)
    {
        CTuint idx = id + offset;

        if(idx >= N)
        {
            break;
        }

        offset += gridDim.x * blockDim.x;

        /*CTuint nodeIndex = eventsSrc.lines[0].nodeIndex[idx];

        BBox bbox = nodesBBoxes[nodeIndex];

        CTuint prefixPrims = nodesContentStart[nodeIndex];

        CTuint primCount = nodesContentCount[nodeIndex];*/

        if(LONGEST_AXIS)
        {
            CTaxis_t axis = (CTaxis_t)getLongestAxis(cache[i].bbox.m_max - cache[i].bbox.m_min);

            CTeventType_t type = eventsSrc.lines[axis].type[idx];

            CTreal split = eventsSrc.lines[axis].indexedEvent[idx].v;

            CTuint startScan = eventsSrc.lines[axis].scannedEventTypeStartMask[idx];
            CTuint endScan = idx - startScan;

//             CTuint above = cache[i].primCount - endScan + cache[i].prefixPrims - type;
//             CTuint below = startScan - cache[i].prefixPrims;

            CTuint above = cache[i].primCount - endScan + cache[i].prefixPrims - (1 ^ type);
            CTuint below = startScan - cache[i].prefixPrims;

            g_splits.above[idx] = above;
            g_splits.below[idx] = below;
            g_splits.indexedSplit[idx].index = idx;

            g_splits.indexedSplit[idx].sah = getSAH(cache[i].bbox, axis, split, below, above);
            g_splits.axis[idx] = axis;
            g_splits.v[idx] = split;
        }
    }

   /* else
    {
        CTreal currentBest = -1;
        for(CTbyte axis = 0; axis < 3; ++axis)
        {
            CTbyte type = eventsSrc.lines[axis].type[id];
            CTreal split = eventsSrc.lines[axis].indexedEvent[id].v;

            CTuint endScan = eventsSrc.lines[axis].scannedEventTypeEndMask[id];
            CTuint startScan = id - endScan;
            CTuint above = primCount - endScan + prefixPrims - type;
            CTuint below = startScan - prefixPrims;
            
            CTreal sah = getSAH(bbox, axis, split, below, above);

            if(currentBest < 0 || sah < currentBest)
            {
                currentBest = sah;
                g_splits.above[id] = above;
                g_splits.below[id] = below;
                g_splits.indexedSplit[id].index = id;

                g_splits.indexedSplit[id].sah = sah;
                g_splits.axis[id] = axis;
                g_splits.v[id] = split;
            }
        }
    } */
}

void cudaComputeSAHSplits3(
    const CTuint* __restrict nodesContentCount,
    const CTuint* __restrict nodesContentStart,
    const BBox* __restrict nodesBBoxes,
    CTuint N,
    CTbyte srcIndex,
    CTuint grid,
    CTuint block,
    cudaStream_t stream) 
{
    computeSAHSplits3<1, 1><<<grid, block, 0, stream>>>(
        nodesContentCount,
        nodesContentStart,
        nodesBBoxes,
        N,
        srcIndex);
}

__device__ __forceinline bool isIn(BBox& bbox, CTaxis_t axis, CTreal v)
{
    return getAxis(bbox.GetMin(), axis) <= v && v <= getAxis(bbox.GetMax(), axis);
}

__device__ __forceinline void copyEvent(cuEventLine dst, cuEventLine src, CTuint dstIndex, CTuint srcIndex)
{
    dst.indexedEvent[dstIndex] = src.indexedEvent[srcIndex];
    //dst.nodeIndex[dstIndex] = src.nodeIndex[srcIndex];
    dst.primId[dstIndex] = src.primId[srcIndex];
    dst.ranges[dstIndex] = src.ranges[srcIndex];
    dst.type[dstIndex] = src.type[srcIndex];
}

__device__ __forceinline void clipCopyEvent(cuEventLine dst, cuEventLine src, CTuint dstIndex, CTuint srcIndex)
{
    dst.indexedEvent[dstIndex] = src.indexedEvent[srcIndex];
    dst.primId[dstIndex] = src.primId[srcIndex];
    dst.type[dstIndex] = src.type[srcIndex];
}

template<CTbyte E_PER_THREAD>
__global__ void clipEventsNT(
    cuEventLineTriple dst,
    cuEventLineTriple src,
    const CTuint* __restrict nodeContentStart,
    const CTuint* __restrict nodeContentCount,
    CTuint eventsPerAxisN,
    CTuint N)
{
    RETURN_IF_OOB(N);

    id = E_PER_THREAD * id;

    CTaxis_t axis = id / eventsPerAxisN;

    id = id % eventsPerAxisN;

// #pragma unroll
// 
//     for(CTbyte axis = 0; axis < 3; ++axis)
//     {

#pragma unroll
        for(CTuint k = 0; k < E_PER_THREAD; ++k)
        {
            CTuint i = id + k;
            if(i >= eventsPerAxisN)
            {
                break;
            }

            CTuint nodeIndex = src.lines[0].nodeIndex[i];
            CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];
            IndexedSAHSplit isplit = g_splitsConst.indexedSplit[eventsLeftFromMe];
            CTaxis_t splitAxis = g_splitsConst.axis[isplit.index];
            CTreal split = g_splitsConst.v[isplit.index];
            BBox bbox = src.lines[axis].ranges[i];
            CTreal v = src.lines[axis].indexedEvent[i].v;    
            CTeventType_t type = src.lines[axis].type[i];

            CTuint N = eventsLeftFromMe + 2 * nodeContentCount[nodeIndex];

            if(getAxis(bbox.m_min, splitAxis) <= split && getAxis(bbox.m_max, splitAxis) <= split)
            {
                dst.lines[axis].mask[eventsLeftFromMe + i] = 1;

                //left
                copyEvent(dst.lines[axis], src.lines[axis], eventsLeftFromMe + i, i);
                dst.lines[axis].nodeIndex[eventsLeftFromMe + i] = 2 * nodeIndex;
            }
            else if(getAxis(bbox.m_min, splitAxis) >= split && getAxis(bbox.m_max, splitAxis) >= split)
            {
                dst.lines[axis].mask[N + i] = 1;

                //right
                copyEvent(dst.lines[axis], src.lines[axis], N + i, i);
                dst.lines[axis].nodeIndex[N + i] = 2 * nodeIndex + 1;
            }
            else
            {
                dst.lines[axis].mask[eventsLeftFromMe + i] = 1;
                dst.lines[axis].mask[N + i] = 1;

                //both
                clipCopyEvent(dst.lines[axis], src.lines[axis], eventsLeftFromMe + i, i);
                CTreal save = getAxis(bbox.m_max, splitAxis);
                setAxis(bbox.m_max, splitAxis, split);
                dst.lines[axis].nodeIndex[eventsLeftFromMe + i] = 2 * nodeIndex;
                dst.lines[axis].ranges[eventsLeftFromMe + i] = bbox;

                setAxis(bbox.m_max, splitAxis, save);
                clipCopyEvent(dst.lines[axis], src.lines[axis], N + i, i);
                setAxis(bbox.m_min, splitAxis, split);
                dst.lines[axis].nodeIndex[N + i] = 2 * nodeIndex+ 1;
                dst.lines[axis].ranges[N + i] = bbox;

                if(axis == splitAxis)
                {
                    CTuint right = !(split > v || (v == split && type == EDGE_END));
                    if(right)
                    {
                        dst.lines[axis].indexedEvent[eventsLeftFromMe + i].v = split;
                    }
                    else
                    {
                        dst.lines[axis].indexedEvent[N + i].v = split;
                    }
                }
            }
        }
    //}
}

__global__ void clipEvents(
    cuEventLine dst,
    cuEventLine src,
    CTuint* mask,
    const CTuint* __restrict nodeContentStart,
    const CTuint* __restrict nodeContentCount,
    CTaxis_t myAxis,
    CTuint eventCount)
{
    RETURN_IF_OOB(eventCount);

    CTuint nodeIndex = src.nodeIndex[id];
    CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];

    IndexedSAHSplit isplit = g_splitsConst.indexedSplit[eventsLeftFromMe];
    
    CTaxis_t splitAxis = g_splitsConst.axis[isplit.index];

    BBox bbox = src.ranges[id];

    CTreal v = src.indexedEvent[id].v;
    CTreal split = g_splitsConst.v[isplit.index];
    CTeventType_t type = src.type[id];

    CTuint N = eventsLeftFromMe + 2 * nodeContentCount[nodeIndex];

    if(getAxis(bbox.m_min, splitAxis) <= split && getAxis(bbox.m_max, splitAxis) <= split)
    {
        mask[eventsLeftFromMe + id] = 1;

        //left
        copyEvent(dst, src, eventsLeftFromMe + id, id);
        dst.nodeIndex[eventsLeftFromMe + id] = 2 * nodeIndex;
    }
    else if(getAxis(bbox.m_min, splitAxis) >= split && getAxis(bbox.m_max, splitAxis) >= split)
    {
        mask[N + id] = 1;

        //right
        copyEvent(dst, src, N + id, id);
        dst.nodeIndex[N + id] = 2 * nodeIndex + 1;
    }
    else
    {
        mask[eventsLeftFromMe + id] = 1;
        mask[N + id] = 1;

        //both
        clipCopyEvent(dst, src, eventsLeftFromMe + id, id);
        CTreal save = getAxis(bbox.m_max, splitAxis);
        setAxis(bbox.m_max, splitAxis, split);
        dst.nodeIndex[eventsLeftFromMe + id] = 2 * nodeIndex;
        dst.ranges[eventsLeftFromMe + id] = bbox;

        setAxis(bbox.m_max, splitAxis, save);
        clipCopyEvent(dst, src, N + id, id);
        setAxis(bbox.m_min, splitAxis, split);
        dst.nodeIndex[N + id] = 2 * nodeIndex + 1;
        dst.ranges[N + id] = bbox;

        if(myAxis == splitAxis)
        {
            CTuint right = !(split > v || (v == split && type == EDGE_END));
            if(right)
            {
                dst.indexedEvent[eventsLeftFromMe + id].v = split;
            }
            else
            {
                dst.indexedEvent[N + id].v = split;
            }
        }
    }
}

__global__ void clipEvents3(
    /*cuEventLineTriple dst,
    cuEventLineTriple src,*/
    const CTuint* __restrict nodeContentStart,
    const CTuint* __restrict nodeContentCount,
    CTuint count,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(count);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    CTuint nodeIndex = eventsSrc.lines[0].nodeIndex[id];
    CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];
    IndexedSAHSplit isplit = g_splitsConst.indexedSplit[eventsLeftFromMe];
    CTaxis_t splitAxis = g_splitsConst.axis[isplit.index];
    CTreal split = g_splitsConst.v[isplit.index];
    CTuint N = eventsLeftFromMe + 2 * nodeContentCount[nodeIndex];

    #pragma unroll
    for(CTaxis_t axis = 0; axis < 3; ++axis)
    {
        BBox bbox = eventsSrc.lines[axis].ranges[id];
        CTreal v = eventsSrc.lines[axis].indexedEvent[id].v;    
        CTeventType_t type = eventsSrc.lines[axis].type[id];

        CTreal minAxis = getAxis(bbox.m_min, splitAxis);
        CTreal maxAxis = getAxis(bbox.m_max, splitAxis);

        if(minAxis <= split && maxAxis <= split)
        {
            eventsDst.lines[axis].mask[eventsLeftFromMe + id] = 1;

            //left
            copyEvent(eventsDst.lines[axis], eventsSrc.lines[axis], eventsLeftFromMe + id, id);
            eventsDst.lines[axis].nodeIndex[eventsLeftFromMe + id] = 2 * nodeIndex;
        }
        else if(minAxis >= split && maxAxis >= split)
        {
            eventsDst.lines[axis].mask[N + id] = 1;

            //right
            copyEvent(eventsDst.lines[axis], eventsSrc.lines[axis], N + id, id);
            eventsDst.lines[axis].nodeIndex[N + id] = 2 * nodeIndex + 1;
        }
        else
        {
            eventsDst.lines[axis].mask[eventsLeftFromMe + id] = 1;
            eventsDst.lines[axis].mask[N + id] = 1;

            //both
            clipCopyEvent(eventsDst.lines[axis], eventsSrc.lines[axis], eventsLeftFromMe + id, id);
            //CTreal save = maxAxis;
            setAxis(bbox.m_max, splitAxis, split);
            eventsDst.lines[axis].nodeIndex[eventsLeftFromMe + id] = 2 * nodeIndex;
            eventsDst.lines[axis].ranges[eventsLeftFromMe + id] = bbox;

            setAxis(bbox.m_max, splitAxis, maxAxis);
            clipCopyEvent(eventsDst.lines[axis], eventsSrc.lines[axis], N + id, id);
            setAxis(bbox.m_min, splitAxis, split);
            eventsDst.lines[axis].nodeIndex[N + id] = 2 * nodeIndex+ 1;
            eventsDst.lines[axis].ranges[N + id] = bbox;

            if(axis == splitAxis)
            {
                CTuint right = !(split > v || (v == split && type == EDGE_END));
                if(right)
                {
                    eventsDst.lines[axis].indexedEvent[eventsLeftFromMe + id].v = split;
                }
                else
                {
                    eventsDst.lines[axis].indexedEvent[N + id].v = split;
                }
            }
        }
    }
}

// __global__ void clearMasks(CTbyte* a, CTbyte* b, CTbyte* c, CTuint N)
// {
//     RETURN_IF_OOB(N);
//     a[id] = 0;
//     b[id] = 0;
//     c[id] = 0;
// }
// 
// __global__ void clearMasks3(CTbyte3* a, CTuint N)
// {
//     RETURN_IF_OOB(N);
//     CTbyte3 v = {0};
//     a[id] = v;
// }

__global__ void createClipMask(
    const CTuint* __restrict nodeContentStart,
    const CTuint* __restrict nodeContentCount,
    CTuint count,
    CTbyte srcIndex, CTuint d = 0)
{
    RETURN_IF_OOB(count);
    EVENT_TRIPLE_HEADER_SRC;

    CTuint nodeIndex = eventsSrc.lines[0].nodeIndex[id];
    CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];
    IndexedSAHSplit isplit = g_splitsConst.indexedSplit[eventsLeftFromMe];

    CTaxis_t splitAxis = g_splitsConst.axis[isplit.index];
    CTreal split = g_splitsConst.v[isplit.index];
    CTuint N = eventsLeftFromMe + 2 * nodeContentCount[nodeIndex];

    CTclipMask_t maskLeft[3] = {0};
    CTclipMask_t maskRight[3] = {0};
    cuClipMask cms[3];

    cms[0] = g_clipArray.mask[0];
    cms[1] = g_clipArray.mask[1];
    cms[2] = g_clipArray.mask[2];
    
#pragma unroll
    for(CTaxis_t axis = 0; axis < 3; ++axis)
    {
        const cuEventLine& __restrict srcLine = eventsSrc.lines[axis];

        BBox& bbox = srcLine.ranges[id];
        CTreal v = srcLine.indexedEvent[id].v;
        CTaxis_t type = srcLine.type[id];

        CTreal minAxis = getAxis(bbox.m_min, splitAxis);
        CTreal maxAxis = getAxis(bbox.m_max, splitAxis);

        const CTreal delta = 0;
        if(maxAxis + delta <= split)
        {
            //left
            setLeft(maskLeft[axis]);
            setAxis(maskLeft[axis], splitAxis);

            cms[axis].index[eventsLeftFromMe + id] = id;
        } 
        else if(minAxis - delta >= split)
        {
            //right
            setRight(maskRight[axis]);
            setAxis(maskRight[axis], splitAxis);

            cms[axis].index[N + id] = id;
        }
        else// if(minAxis < split && split < maxAxis)
        {
            cms[axis].index[N + id] = id;
            cms[axis].newSplit[N + id] = split;
            cms[axis].newSplit[eventsLeftFromMe + id] = split;
            cms[axis].index[eventsLeftFromMe + id] = id;

            //CTbyte mar = 0;
            //both
            setLeft(maskLeft[axis]);
            setAxis(maskLeft[axis], splitAxis);
            
            setRight(maskRight[axis]);
            setAxis(maskRight[axis], splitAxis);

            setOLappin(maskLeft[axis]);
            setOLappin(maskRight[axis]);
   
            if(axis == splitAxis)
            {
                CTuint right = !(split > v || (v == split && type == EDGE_END));
                if(right)
                {
                    maskLeft[axis] |= 0x40;
                }
                else
                {
                    maskRight[axis] |= 0x40;
                }
            }
        }
    }

#pragma unroll
    for(CTaxis_t axis = 0; axis < 3; ++axis)
    {
        cms[axis].mask[eventsLeftFromMe + id] = maskLeft[axis];
        cms[axis].mask[N + id] = maskRight[axis];
    }
}

void cudaClipEventsMask(
    const CTuint* __restrict nodeContentStart,
    const CTuint* __restrict nodeContentCount,
    CTuint count,
    CTbyte srcIndex, CTuint grid, CTuint block, cudaStream_t stream)
{
    clipEvents3<<<grid, block, 0, stream>>>(nodeContentStart, nodeContentCount, count, srcIndex);
}

void cudaCreateClipMask(
    const CTuint* __restrict nodeContentStart,
    const CTuint* __restrict nodeContentCount,
    CTuint count,
    CTbyte srcIndex, CTuint grid, CTuint block, cudaStream_t stream)
{
    createClipMask<<<grid, block, 0, stream>>>(nodeContentStart, nodeContentCount, count, srcIndex);
}

__device__ int getWarpCompactPrefix(int mask, int laneId, uint* warpPrefix)
{
    uint bitMask = __ballot(mask);
    *warpPrefix = __popc(bitMask & ((1 << laneId) - 1));
    return __popc(bitMask);
}

template<typename T>
__device__ int warpCompact(int mask, T data, T* result, int laneId)
{
    uint bitMask = __ballot(mask);
    uint warpPrefix = __popc(bitMask & ((1 << laneId) - 1));
    result[warpPrefix] = data;
    return __popc(bitMask);
}

__global__ void countValidElementsPerTile(
    CTuint tilesPerWarp,
    CTuint N)
{   
    CTuint id = GlobalId;
    int warpId = id / warpSize;
    int laneId = id % warpSize;

#pragma unroll
    for(int axis = 0; axis < 3; ++axis)
    {
        CTuint elemCount = 0;
        for(int i = 0; i < tilesPerWarp; ++i)
        {
            int addr = warpId * tilesPerWarp * warpSize + i * warpSize + laneId;
            CTclipMask_t m = addr >= N ? 0 : cms[axis].mask[addr];
            int bitMask = __ballot(m);
            elemCount += __popc(bitMask);
        }

        if(!laneId && warpId < N)
        {
            cms[axis].elemsPerTile[warpId] = elemCount;
        }
    }
}

template<int blockSize>
__global__ void countValidElementsPerTile2(
    CTuint tilesPerWarp,
    CTuint N)
{   
    int warpId = blockIdx.x;
    int laneId = threadIdx.x;
    __shared__ CTuint s_scanned[blockSize/32];
#pragma unroll
    for(int axis = 0; axis < 3; ++axis)
    {
        CTuint elemCount = 0;
        for(int i = 0; i < tilesPerWarp; ++i)
        {
            int addr = warpId * tilesPerWarp * blockSize + i * blockSize + laneId;
            CTclipMask_t mask = addr >= N ? 0 : cms[axis].mask[addr];
            __blockBinaryPrefixSums(s_scanned, mask > 0);
            elemCount += s_scanned[blockSize/32];
        }

        if(!laneId && warpId < N)
        {
            cms[axis].elemsPerTile[warpId] = elemCount;
        }
    }
}

template<int lanesPerBlock>
__global__ void compactEventLineV21(
    CTbyte srcIndex,
    CTuint tilesPerLanes,
    CTuint activeLanes,
    CTuint N)
{
    const int laneSize = 32;
    __shared__ IndexedEvent s_indexEvents[laneSize * lanesPerBlock];
    __shared__ BBox s_bboxes[laneSize * lanesPerBlock];
    __shared__ CTeventType_t s_eventTypes[laneSize * lanesPerBlock];
    __shared__ CTuint s_primIds[laneSize * lanesPerBlock];
    __shared__ CTuint s_nodeIndices[laneSize * lanesPerBlock];

    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    CTuint id = threadIdx.x + blockDim.x * blockIdx.x;
    
    int laneId = threadIdx.x % laneSize;
    int warpId = id / laneSize;

    if(warpId >= activeLanes)
    {
        return;
    }

    IndexedEvent e;
    BBox bbox;
    CTuint primId;
    CTeventType_t type;
    CTuint nnodeIndex;

    CTuint warpPrefix;

    CTuint shrdMemOffset = (threadIdx.x / laneSize) * laneSize;

#pragma unroll
    for(int axis = 0; axis < 3; ++axis)
    {
        CTuint buffered = 0;
        if(cms[axis].elemsPerTile[warpId] == 0)
        {
            continue;
        }
        CTuint tileOffset = cms[axis].scanned[warpId];
        int elemCount;
        for(int i = 0; i < tilesPerLanes; ++i)
        {
            int addr = laneSize * (warpId * lanesPerBlock + i) + laneId;
            CTclipMask_t mask = (addr >= N ? 0 : cms[axis].mask[addr]);
            bool right = isRight(mask);
            elemCount = getWarpCompactPrefix(mask, laneId, &warpPrefix);

            if(mask)
            {
                CTuint eventIndex = cms[axis].index[addr];
                e = eventsSrc.lines[axis].indexedEvent[eventIndex];
                bbox = eventsSrc.lines[axis].ranges[eventIndex];
                primId = eventsSrc.lines[axis].primId[eventIndex];
                type = eventsSrc.lines[axis].type[eventIndex];

                if(axis == 0)
                    nnodeIndex = 2 * eventsSrc.lines[axis].nodeIndex[eventIndex] + (CTuint)right;

                CTaxis_t splitAxis = getAxisFromMask(mask);

                if(isOLappin(mask))
                {
                    CTreal split = cms[axis].newSplit[addr];
                    setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
                    if(i == splitAxis && ((mask & 0x40) == 0x40))
                    {
                        e.v = split;
                    }
                }
                
                if(warpPrefix + buffered < laneSize)
                {
                    *(s_indexEvents + warpPrefix + buffered) = e;
                    *(s_primIds     + warpPrefix + buffered) = primId;
                    *(s_bboxes      + warpPrefix + buffered) = bbox;
                    *(s_eventTypes  + warpPrefix + buffered) = type;

                    if(axis == 0)
                    {
                        *(s_nodeIndices + warpPrefix + buffered) = nnodeIndex;
                    }
                }
            }

           if(buffered + elemCount >= laneSize)
            {
                eventsDst.lines[axis].indexedEvent[tileOffset + laneId] = s_indexEvents[laneId];
                eventsDst.lines[axis].primId      [tileOffset + laneId] = s_primIds    [laneId];
                eventsDst.lines[axis].ranges      [tileOffset + laneId] = s_bboxes     [laneId];
                eventsDst.lines[axis].type        [tileOffset + laneId] = s_eventTypes [laneId];

                if(axis == 0)
                {
                    eventsDst.lines[axis].nodeIndex[tileOffset + laneId] = s_nodeIndices[laneId];
                }

                tileOffset += laneSize;
            }

            if(buffered + warpPrefix >= laneSize &&  mask)
            {
                *(s_indexEvents + warpPrefix + buffered - laneSize) = e;
                *(s_primIds     + warpPrefix + buffered - laneSize) = primId;
                *(s_bboxes      + warpPrefix + buffered - laneSize) = bbox;
                *(s_eventTypes  + warpPrefix + buffered - laneSize) = type;

                if(axis == 0)
                {
                    *(s_nodeIndices + warpPrefix + buffered - laneSize) = nnodeIndex;
                }
            }

            buffered = (buffered + elemCount) % laneSize;
        }

        if(laneId < buffered)
        {
            eventsDst.lines[axis].indexedEvent[tileOffset + laneId] = s_indexEvents[laneId];
            eventsDst.lines[axis].primId      [tileOffset + laneId] = s_primIds    [laneId];
            eventsDst.lines[axis].ranges      [tileOffset + laneId] = s_bboxes     [laneId];
            eventsDst.lines[axis].type        [tileOffset + laneId] = s_eventTypes [laneId];

            if(axis == 0)
            {
                eventsDst.lines[axis].nodeIndex[tileOffset + laneId] = s_nodeIndices[laneId];
            }
        }
    }
}


template<int blockSize>
__global__ void optimizedcompactEventLineV31(
    CTbyte srcIndex,
    CTuint tilesPerLanes,
    CTuint activeLanes,
    CTuint N)
{
    __shared__ IndexedEvent s_indexEvents[blockSize];
    __shared__ BBox s_bboxes[blockSize];
    __shared__ CTeventType_t s_eventTypes[blockSize];
    __shared__ CTuint s_primIds[blockSize];
    __shared__ CTuint s_nodeIndices[blockSize];
    __shared__ CTuint s_scanned[blockSize/32];

    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    int laneId = threadIdx.x;
    int warpId = blockIdx.x;

    if(warpId >= activeLanes)
    {
        return;
    }

    IndexedEvent e;
    BBox bbox;
    CTuint primId;
    CTeventType_t type;
    CTuint nnodeIndex;

    CTuint blockPrefix;

    CTuint shrdMemOffset = 0;

#pragma unroll
    for(int axis = 0; axis < 3; ++axis)
    {
        CTuint buffered = 0;

        if(cms[axis].elemsPerTile[warpId] == 0)
        {
            continue;
        }

        CTuint tileOffset = cms[axis].scanned[warpId];
        int elemCount;

        for(int i = 0; i < tilesPerLanes; ++i)
        {
            int addr = blockSize * tilesPerLanes * warpId + blockSize * i + laneId;
            CTclipMask_t mask = (addr >= N ? 0 : cms[axis].mask[addr]);
            bool right = isRight(mask);

            blockPrefix = __blockBinaryPrefixSums(s_scanned, mask > 0);

            elemCount = s_scanned[blockSize/32];

            if(mask)
            {
                CTuint eventIndex = cms[axis].index[addr];
                e = eventsSrc.lines[axis].indexedEvent[eventIndex];
                bbox = eventsSrc.lines[axis].ranges[eventIndex];
                primId = eventsSrc.lines[axis].primId[eventIndex];
                type = eventsSrc.lines[axis].type[eventIndex];

                if(axis == 0)
                    nnodeIndex = 2 * eventsSrc.lines[axis].nodeIndex[eventIndex] + (CTuint)right;

                CTaxis_t splitAxis = getAxisFromMask(mask);

                if(isOLappin(mask))
                {
                    CTreal split = cms[axis].newSplit[addr];
                    setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
                    if(i == splitAxis && ((mask & 0x40) == 0x40))
                    {
                        e.v = split;
                    }
                }

                if(blockPrefix + buffered < blockSize)
                {
                    *(s_indexEvents + blockPrefix + buffered) = e;
                    *(s_primIds     + blockPrefix + buffered) = primId;
                    *(s_bboxes      + blockPrefix + buffered) = bbox;
                    *(s_eventTypes  + blockPrefix + buffered) = type;

                    if(axis == 0)
                    {
                        *(s_nodeIndices + blockPrefix + buffered) = nnodeIndex;
                    }
                }
            }

             __syncthreads();

            if(buffered + elemCount >= blockSize)
            {
                eventsDst.lines[axis].indexedEvent[tileOffset + laneId] = s_indexEvents[laneId];
                eventsDst.lines[axis].primId      [tileOffset + laneId] = s_primIds    [laneId];
                eventsDst.lines[axis].ranges      [tileOffset + laneId] = s_bboxes     [laneId];
                eventsDst.lines[axis].type        [tileOffset + laneId] = s_eventTypes [laneId];

                if(axis == 0)
                {
                    eventsDst.lines[axis].nodeIndex[tileOffset + laneId] = s_nodeIndices[laneId];
                }

                tileOffset += blockSize;
            }

             __syncthreads();

            if(buffered + blockPrefix >= blockSize &&  mask)
            {
                *(s_indexEvents + blockPrefix + buffered - blockSize) = e;
                *(s_primIds     + blockPrefix + buffered - blockSize) = primId;
                *(s_bboxes      + blockPrefix + buffered - blockSize) = bbox;
                *(s_eventTypes  + blockPrefix + buffered - blockSize) = type;

                if(axis == 0)
                {
                    *(s_nodeIndices + blockPrefix + buffered - blockSize) = nnodeIndex;
                }
            }

            buffered = (buffered + elemCount) % blockSize;
        }

        __syncthreads();

        if(laneId < buffered)
        {
            eventsDst.lines[axis].indexedEvent[tileOffset + laneId] = s_indexEvents[laneId];
            eventsDst.lines[axis].primId      [tileOffset + laneId] = s_primIds    [laneId];
            eventsDst.lines[axis].ranges      [tileOffset + laneId] = s_bboxes     [laneId];
            eventsDst.lines[axis].type        [tileOffset + laneId] = s_eventTypes [laneId];

            if(axis == 0)
            {
                eventsDst.lines[axis].nodeIndex[tileOffset + laneId] = s_nodeIndices[laneId];
            }
        }
    }
}

__global__ void compactEventLineV2(
    CTuint N,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    CTuint masks[3];
    masks[0] = cms[0].mask[id];
    masks[1] = cms[1].mask[id];
    masks[2] = cms[2].mask[id];

    #pragma unroll
    for(CTaxis_t i = 0; i < 3; ++i)
    {
        if(isSet(masks[i]))
        {
            CTuint eventIndex = cms[i].index[id];

            CTuint dstAdd = cms[i].scanned[id];

            IndexedEvent e = eventsSrc.lines[i].indexedEvent[eventIndex];
            BBox bbox = eventsSrc.lines[i].ranges[eventIndex];
            CTuint primId = eventsSrc.lines[i].primId[eventIndex];
            CTeventType_t type = eventsSrc.lines[i].type[eventIndex];

            CTaxis_t splitAxis = getAxisFromMask(masks[i]);
            bool right = isRight(masks[i]);

            CTuint nnodeIndex;

            if(i == 0)
            {
                nnodeIndex = eventsSrc.lines[i].nodeIndex[eventIndex];
            }

            if(isOLappin(masks[i]))
            {
                CTreal split = cms[i].newSplit[id];
                setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
                if(i == splitAxis && ((masks[i] & 0x40) == 0x40))
                {
                    e.v = split;
                }
            }

            eventsDst.lines[i].indexedEvent[dstAdd] = e;
            eventsDst.lines[i].primId[dstAdd] = primId;
            eventsDst.lines[i].ranges[dstAdd] = bbox;
            eventsDst.lines[i].type[dstAdd] = type;

            if(i == 0)
            {
                eventsDst.lines[i].nodeIndex[dstAdd] = 2 * nnodeIndex + (CTuint)right;
            }
        }
    }
}

__global__ void initInteriorNodes(
    const CTuint* __restrict activeNodes,
    const CTuint* __restrict activeNodesThisLevel,
    const BBox* __restrict oldBBoxes,
    BBox* newBBoxes,
    CTuint* contenCount,
    CTuint* newContentCount,
    CTuint* newActiveNodes,
    CTnodeIsLeaf_t* activeNodesIsLeaf,
    CTuint childOffset,
    CTuint nodeOffset,
    CTuint N,
    CTuint* oldNodeContentStart,
    CTnodeIsLeaf_t* gotLeaves,
    CTbyte makeLeaves,
    CTbyte _gotLeafes)
{
    RETURN_IF_OOB(N);

    CTuint can;
    if(_gotLeafes)
    {
        can = activeNodesThisLevel[id];
    }
    else
    {
        can = id;
    }

    CTuint an = activeNodes[id];
    CTuint edgesBeforeMe = 2 * oldNodeContentStart[can];

    IndexedSAHSplit split = g_splitsConst.indexedSplit[edgesBeforeMe];

    CTaxis_t axis = g_splitsConst.axis[split.index];
    CTreal scanned = g_splitsConst.v[split.index];

    CTuint below = g_splitsConst.below[split.index];
    CTuint above = g_splitsConst.above[split.index];

    CTuint nodeId = nodeOffset + an;
    g_nodes.split[nodeId] = scanned;
    g_nodes.splitAxis[nodeId] = axis;

    CTuint dst = id;

    newContentCount[2 * dst + 0] = below;
    newContentCount[2 * dst + 1] = above;

    CTuint leftChildIndex = childOffset + 2 * can + 0;
    CTuint rightChildIndex = childOffset + 2 * can + 1;

    g_nodes.leftChild[nodeId] = leftChildIndex;
    g_nodes.rightChild[nodeId] = rightChildIndex;

    if((makeLeaves || (below <= MAX_ELEMENTS_PER_LEAF || above <= MAX_ELEMENTS_PER_LEAF)))
    {
        gotLeaves[0] = 1;
    }

    g_nodes.isLeaf[childOffset + 2 * can + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
    g_nodes.isLeaf[childOffset + 2 * can + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;

    activeNodesIsLeaf[2 * id + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
    activeNodesIsLeaf[2 * id + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
    newActiveNodes[2 * id + 0] = 2 * can + 0;
    newActiveNodes[2 * id + 1] = 2 * can + 1;

    BBox l;
    BBox r;

    splitAABB(&oldBBoxes[id], scanned, axis, &l, &r);

    if(below > MAX_ELEMENTS_PER_LEAF)
    {
        newBBoxes[2 * dst + 0] = l;
    }

    if(above > MAX_ELEMENTS_PER_LEAF)
    {
        newBBoxes[2 * dst + 1] = r;
    }
}

void cudaInitInteriorNodes(
    const CTuint* __restrict activeNodes,
    const CTuint* __restrict activeNodesThisLevel,
    const BBox* __restrict oldBBoxes,
    BBox* newBBoxes,
    CTuint* contenCount,
    CTuint* newContentCount,
    CTuint* newActiveNodes,
    CTnodeIsLeaf_t* activeNodesIsLeaf,
    CTuint childOffset,
    CTuint nodeOffset,
    CTuint N,
    CTuint* oldNodeContentStart,
    CTnodeIsLeaf_t* gotLeaves,
    CTbyte makeLeaves,
    CTbyte _gotLeafes,
    CTuint grid, CTuint block, cudaStream_t stream)
{
    initInteriorNodes<<<grid, block, 0, stream>>>(
        activeNodes,
        activeNodesThisLevel,

        oldBBoxes, 
        newBBoxes, 

        contenCount,

        newContentCount,
        newActiveNodes,
        activeNodesIsLeaf,

        childOffset,
        nodeOffset,
        N,
        oldNodeContentStart,
        gotLeaves,
        makeLeaves,
        _gotLeafes);
}

__global__ void compactEventLineV4(
    CTuint N,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    int i = id / (N / 3);

    id = id % (N / 3);

    CTuint masks = cms[i].mask[id];
    CTaxis_t splitAxis = getAxisFromMask(masks);

    if(isSet(masks))
    {
        CTuint eventIndex = cms[i].index[id];

        CTuint dstAdd = cms[i].scanned[id];

        IndexedEvent e = eventsSrc.lines[i].indexedEvent[eventIndex];
        BBox bbox = eventsSrc.lines[i].ranges[eventIndex];
        CTuint primId = eventsSrc.lines[i].primId[eventIndex];
        CTeventType_t type = eventsSrc.lines[i].type[eventIndex];

        bool right = isRight(masks);

        CTuint nnodeIndex;

        if(i == 0)
        {
            nnodeIndex = eventsSrc.lines[i].nodeIndex[eventIndex];
        }

        if(isOLappin(masks))
        {
            CTreal split = cms[i].newSplit[id];
            setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
            if(i == splitAxis && ((masks & 0x40) == 0x40))
            {
                e.v = split;
            }
        }

        eventsDst.lines[i].indexedEvent[dstAdd] = e;
        eventsDst.lines[i].primId[dstAdd] = primId;
        eventsDst.lines[i].ranges[dstAdd] = bbox;
        eventsDst.lines[i].type[dstAdd] = type;

        if(i == 0)
        {
            eventsDst.lines[i].nodeIndex[dstAdd] = 2 * nnodeIndex + (CTuint)right;
        }
    }
}

void cudaCompactEventLineV2(
    CTuint N,
    CTbyte srcIndex, CTuint grid, CTuint block, cudaStream_t stream)
{
    compactEventLineV4<<<grid, block, 0, stream>>>(
        N,
        srcIndex);

//     compactEventLineV2<<<grid, block, 0, stream>>>(
//         N,
//         srcIndex);
}

template<int blockSize, uint compactLineSize>
__global__ void compactEventLineV2Buffered(
    CTbyte srcIndex,
    CTuint N)
{
    __shared__ IndexedEvent s_indexEvents[blockSize];
    __shared__ BBox s_bboxes[blockSize];
    __shared__ CTeventType_t s_eventTypes[blockSize];
    __shared__ CTuint s_primIds[blockSize];
    __shared__ CTuint s_nodeIndices[blockSize];
    __shared__ CTuint s_scanned[blockSize];

    s_nodeIndices[threadIdx.x] = -1;

    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    IndexedEvent e;
    BBox bbox;
    CTuint primId;
    CTeventType_t type;
    CTuint nnodeIndex;

    CTuint blockPrefix;

#pragma unroll
    for(int axis = 0; axis < 3; ++axis)
    {
        CTuint buffered = 0;

        CTuint tileOffset = cms[axis].scanned[blockIdx.x * compactLineSize];
        int elemCount;

        for(int i = 0; i < compactLineSize; i += blockSize)
        {
            uint addr = blockIdx.x * compactLineSize + threadIdx.x + i;
            CTclipMask_t mask = (addr >= N ? 0 : cms[axis].mask[addr]);
            bool right = isRight(mask);

            blockPrefix = __blockBinaryPrefixSums(s_scanned, mask > 0);

            elemCount = s_scanned[blockSize/32];

            if(mask)
            {
                CTuint eventIndex = cms[axis].index[addr];
                e = eventsSrc.lines[axis].indexedEvent[eventIndex];
                bbox = eventsSrc.lines[axis].ranges[eventIndex];
                primId = eventsSrc.lines[axis].primId[eventIndex];
                type = eventsSrc.lines[axis].type[eventIndex];

                if(axis == 0)
                    nnodeIndex = 2 * eventsSrc.lines[axis].nodeIndex[eventIndex] + (CTuint)right;

                CTaxis_t splitAxis = getAxisFromMask(mask);

                if(isOLappin(mask))
                {
                    CTreal split = cms[axis].newSplit[addr];
                    setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
                    if(i == splitAxis && ((mask & 0x40) == 0x40))
                    {
                        e.v = split;
                    }
                }

                if(blockPrefix + buffered < blockSize)
                {
                    *(s_indexEvents + blockPrefix + buffered) = e;
                    *(s_primIds     + blockPrefix + buffered) = primId;
                    *(s_bboxes      + blockPrefix + buffered) = bbox;
                    *(s_eventTypes  + blockPrefix + buffered) = type;

                    if(axis == 0)
                    {
                        *(s_nodeIndices + blockPrefix + buffered) = nnodeIndex;
                    }
                }
            }

            __syncthreads();

            if(buffered + elemCount >= blockSize)
            {
                eventsDst.lines[axis].indexedEvent[tileOffset + threadIdx.x] = s_indexEvents[threadIdx.x];
                eventsDst.lines[axis].primId      [tileOffset + threadIdx.x] = s_primIds    [threadIdx.x];
                eventsDst.lines[axis].ranges      [tileOffset + threadIdx.x] = s_bboxes     [threadIdx.x];
                eventsDst.lines[axis].type        [tileOffset + threadIdx.x] = s_eventTypes [threadIdx.x];

                if(axis == 0)
                {
                    CTuint id = s_nodeIndices[threadIdx.x];
                    eventsDst.lines[axis].nodeIndex[tileOffset + threadIdx.x] = id;
                }

                tileOffset += blockSize;
            }

            __syncthreads();

            if(buffered + blockPrefix >= blockSize &&  mask)
            {
               *(s_indexEvents + blockPrefix + buffered - blockSize) = e;
               *(s_primIds     + blockPrefix + buffered - blockSize) = primId;
               *(s_bboxes      + blockPrefix + buffered - blockSize) = bbox;
               *(s_eventTypes  + blockPrefix + buffered - blockSize) = type;

                if(axis == 0)
                {
                    *(s_nodeIndices + blockPrefix + buffered - blockSize) = nnodeIndex;
                }
            }

            buffered = (buffered + elemCount) % blockSize;
        }

        __syncthreads();

        if(threadIdx.x < buffered)
        {
            eventsDst.lines[axis].indexedEvent[tileOffset + threadIdx.x] = s_indexEvents[threadIdx.x];
            eventsDst.lines[axis].primId      [tileOffset + threadIdx.x] = s_primIds    [threadIdx.x];
            eventsDst.lines[axis].ranges      [tileOffset + threadIdx.x] = s_bboxes     [threadIdx.x];
            eventsDst.lines[axis].type        [tileOffset + threadIdx.x] = s_eventTypes [threadIdx.x];

            if(axis == 0)
            {
                CTuint id = s_nodeIndices[threadIdx.x];
                eventsDst.lines[axis].nodeIndex[tileOffset + threadIdx.x] = id;
            }
        }
    }
}

void cudaCompactEventLineV2Buffered(
    CTuint N,
    CTbyte srcIndex, CTuint grid, cudaStream_t stream)
{
    const uint blockSize = 256;
    const uint lineSize = 2*blockSize; //8*blockSize;
    grid = nutty::cuda::GetCudaGrid(N,  lineSize);
    compactEventLineV2Buffered<blockSize, lineSize><<<grid, blockSize, 0, stream>>>(
        srcIndex, N);
}

__global__ void setEventsBelongToLeafAndSetNodeIndex(
    const CTnodeIsLeaf_t* __restrict isLeaf,
    CTeventIsLeaf_t* eventIsLeaf,
    CTuint* nodeToLeafIndex,
    CTuint N,
    CTuint N2,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;

    //g_splitsConst.indexedSplit

    eventIsLeaf[id] = isLeaf[eventsSrc.lines[0].nodeIndex[id]] && (eventsSrc.lines[0].type[id] == EVENT_START);

    if(id < N2)
    {
       nodeToLeafIndex[id] = 0;
    }
}

void cudaSetEventsBelongToLeafAndSetNodeIndex(
    const CTnodeIsLeaf_t* __restrict isLeaf,
    CTeventIsLeaf_t* eventIsLeaf,
    CTuint* nodeToLeafIndex,
    CTuint N,
    CTuint N2,
    CTbyte srcIndex, CTuint grid, CTuint block, cudaStream_t stream)
{
    setEventsBelongToLeafAndSetNodeIndex<<<grid, block, 0, stream>>>(
        isLeaf,
        eventIsLeaf,
        nodeToLeafIndex,
        N,
        N2,
        srcIndex);
}

__global__ void compactLeafData(
    CTuint* leafContent,
    const CTuint* __restrict leafContentCount,
    const CTuint* __restrict leafContentStart, 
    const CTeventIsLeaf_t* __restrict eventIsLeafMask, 
    const CTuint* __restrict leafEventScanned,
    CTuint currentLeafCount,
    CTbyte srcIndex,
    CTuint N)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_DST;
    
    CTuint offset = 0;
    if(currentLeafCount)
    {
        offset = leafContentStart[currentLeafCount-1] + leafContentCount[currentLeafCount-1];
    }

    if(eventIsLeafMask[id])
    {
        leafContent[offset + leafEventScanned[id]] = eventsDst.lines[0].primId[id];
    }
}

struct TmpcuEventLine
{
    IndexedEvent indexedEvent;
    CTeventType_t type;
    CTuint nodeIndex;
    CTuint primId;
    BBox ranges;
    CTeventMask_t mask;
};

template<CTuint packNodes>
__global__ void compactMakeLeavesData(
    const CTnodeIsLeaf_t* __restrict activeNodeIsLeaf,
    //const CTuint* __restrict interiorCountScanned,
    //const CTuint* __restrict scannedLeafContent,
    const CTuint* __restrict nodesContentScanned,
    const CTuint* __restrict leafEventScanned,

    const CTuint* __restrict nodesContentCount,
    const CTeventIsLeaf_t* __restrict eventIsLeafMask,

    const CTuint* __restrict leafCountScan, 

    const CTuint* __restrict activeNodes,
    const CTuint* __restrict isLeafScan,
    const CTuint* __restrict scannedInteriorContent,
    const BBox* __restrict nodeBBoxes,

    CTuint* leafContent,

    CTuint* nodeToLeafIndex,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* leafContentStart, 
    CTuint* leafContentCount,
    CTuint* newActiveNodes,
    BBox* newBBxes,
    
    CTuint offset,
    CTuint leafContentStartOffset,
    CTuint currentLeafCount,
    CTuint nodeCount,
    CTbyte srcIndex,
    CTuint N,
    CTuint d = 0)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    CTuint oldNodeIndex = eventsSrc.lines[0].nodeIndex[id];

    if(!activeNodeIsLeaf[oldNodeIndex])
    {

        CTuint eventsBeforeMe = 2 * (nodesContentScanned[oldNodeIndex] - scannedInteriorContent[oldNodeIndex]);//scannedLeafContent[oldNodeIndex];
        CTuint nodeIndex = oldNodeIndex - leafCountScan[oldNodeIndex]; //interiorCountScanned[oldNodeIndex];

        eventsDst.lines[0].nodeIndex[id - eventsBeforeMe] = nodeIndex;
                
        TmpcuEventLine lines[3];
        lines[0].indexedEvent = eventsSrc.lines[0].indexedEvent[id];
        lines[1].indexedEvent = eventsSrc.lines[1].indexedEvent[id];
        lines[2].indexedEvent = eventsSrc.lines[2].indexedEvent[id];

        lines[0].primId = eventsSrc.lines[0].primId[id];
        lines[1].primId = eventsSrc.lines[1].primId[id];
        lines[2].primId = eventsSrc.lines[2].primId[id];

        lines[0].ranges = eventsSrc.lines[0].ranges[id];
        lines[1].ranges = eventsSrc.lines[1].ranges[id];
        lines[2].ranges = eventsSrc.lines[2].ranges[id];

        lines[0].type = eventsSrc.lines[0].type[id];
        lines[1].type = eventsSrc.lines[1].type[id];
        lines[2].type = eventsSrc.lines[2].type[id];

        #pragma unroll
        for(CTaxis_t i = 0; i < 3; ++i)
        {
            //const cuEventLine& __restrict srcLine = eventsSrc.lines[i];
            cuEventLine& dstLine = eventsDst.lines[i];
                    
            dstLine.indexedEvent[id - eventsBeforeMe] = lines[i].indexedEvent;
            dstLine.primId[id - eventsBeforeMe] = lines[i].primId;
            dstLine.ranges[id - eventsBeforeMe] = lines[i].ranges;
            dstLine.type[id - eventsBeforeMe] = lines[i].type;
        }            
    }
    else if(eventIsLeafMask[id])
    {
        leafContent[leafContentStartOffset + leafEventScanned[id]] = eventsSrc.lines[0].primId[id];
    }

    if(packNodes)
    {
        if(id < nodeCount)
        {
            CTnodeIsLeaf_t leafMask = activeNodeIsLeaf[id];
            CTuint dst = leafCountScan[id];
            if(leafMask && leafMask < 2)
            {
                leafContentStart[currentLeafCount + dst] = leafContentStartOffset + (nodesContentScanned[id] - scannedInteriorContent[id]); //scannedLeafContent[id];
                leafContentCount[currentLeafCount + dst] = nodesContentCount[id];
                nodeToLeafIndex[activeNodes[id] + offset] = currentLeafCount + isLeafScan[id];
            }
            else
            {
                dst = id - dst;//leafCountScan[id]; //interiorCountScanned[id];
                newActiveNodes[dst] = activeNodes[id];
                newContentCount[dst] = nodesContentCount[id];
                newContentStart[dst] = scannedInteriorContent[id];
                newBBxes[dst] = nodeBBoxes[id];
            }
        }
    }
}

void cudaCompactMakeLeavesData(

    const CTnodeIsLeaf_t* __restrict activeNodeIsLeaf,
    const CTuint* __restrict nodesContentScanned,
    const CTuint* __restrict leafEventScanned,

    const CTuint* __restrict nodesContentCount,
    const CTeventIsLeaf_t* __restrict eventIsLeafMask,

    const CTuint* __restrict leafCountScan, 

    const CTuint* __restrict activeNodes,
    const CTuint* __restrict isLeafScan,
    const CTuint* __restrict scannedInteriorContent,
    const BBox* __restrict nodeBBoxes,

    CTuint* leafContent,

    CTuint* nodeToLeafIndex,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* leafContentStart, 
    CTuint* leafContentCount,
    CTuint* newActiveNodes,
    BBox* newBBxes,

    CTuint offset,
    CTuint leafContentStartOffset,
    CTuint currentLeafCount,
    CTuint nodeCount,
    CTbyte srcIndex,
    CTuint N,
    CTuint grid, CTuint block, cudaStream_t stream)
{
    compactMakeLeavesData<1><<<grid, block, 0, stream>>>(
        activeNodeIsLeaf,
        nodesContentScanned,

        leafEventScanned,

        nodesContentCount,
        eventIsLeafMask,

        leafCountScan, 

        activeNodes,
        isLeafScan,
        scannedInteriorContent,
        nodeBBoxes,

        leafContent,
        nodeToLeafIndex,
        newContentCount,
        newContentStart,
        leafContentStart,
        leafContentCount,
        newActiveNodes,
        newBBxes,

        offset,
        leafContentStartOffset,
        currentLeafCount,
        nodeCount,
        srcIndex,
        N);
}

__global__ void make3AxisType(CTbyte3* dst, cuEventLineTriple events, CTuint N)
{
    RETURN_IF_OOB(N);
    CTbyte3 val;
    val.x = events.lines[0].type[id];
    val.y = events.lines[0].type[id];
    val.z = events.lines[0].type[id];
    dst[id] = val;
}

__global__ void createInteriorContentCountMasks(
    const CTnodeIsLeaf_t* __restrict isLeaf,
    const CTuint* __restrict contentCount,
    CTuint* interiorMask,
    CTuint N)
{
    RETURN_IF_OOB(N);
    CTnodeIsLeaf_t mask = isLeaf[id];
    interiorMask[id] = (mask < 2) * (1 ^ mask) * contentCount[id];
}

__device__ CTuint d_nodeOffset;
__device__ CTnodeIsLeaf_t* d_activeNodesIsLeaf;

struct InteriorMaskOp
{
    __device__ CTuint operator()(CTuint elem, CTuint index)
    {
        CTnodeIsLeaf_t mask = d_activeNodesIsLeaf[d_nodeOffset + index];
        return (mask == 0) * elem;
    }

    __device__ CTuint GetNeutral(void)
    {
        return 0;
    }
};

void cudaCreateInteriorContentCountMasks(
    const CTnodeIsLeaf_t* __restrict isLeaf,
    const CTuint* __restrict contentCount,
    CTuint* interiorMask,
    CTuint N, CTuint grid, CTuint block, cudaStream_t stream)
{
    createInteriorContentCountMasks<<<grid, block, 0, stream>>>(
        isLeaf,
        contentCount, 
        interiorMask, N);
}

__device__ __forceinline__ IndexedSAHSplit ris(IndexedSAHSplit t0, IndexedSAHSplit t1)
{
    return t0.sah < t1.sah ? t0 : t1;
}

// __device__ int shfl_add(int x, int offset, int width = warpSize) {
//     int result = 0;
// #if __CUDA_ARCH__ >= 300
//     int mask = (WARP_SIZE - width)<< 8;
//     asm(
//         "{.reg .s32 r0;"
//         ".reg .pred p;"
//         "shfl.up.b32 r0|p, %1, %2, %3;"
//         "@p add.s32 r0, r0, %4;"
//         "mov.s32 %0, r0; }"
//         : "=r"(result) : "r"(x), "r"(offset), "r"(mask), "r"(x));
// #endif
//     return result;
// }
// 
// __device__ int Reduce(int tid, int x, IndexedSAHSplit* storage, int NT) 
// {
//     const int NumSections = warpSize;
//     const int SecSize = NT / NumSections;
//     int lane = (SecSize - 1) & tid;
//     int sec = tid / SecSize;
// 
//     // In the first phase, threads cooperatively find the reduction within
//     // their segment. The segments are SecSize threads (NT / WARP_SIZE) 
//     // wide.
// #pragma unroll
//     for(int offset = 1; offset < SecSize; offset *= 2)
//         x = shfl_add(x, offset, SecSize);
// 
//     // The last thread in each segment stores the local reduction to shared
//     // memory.
//     if(SecSize - 1 == lane) storage[sec] = x;
//     __syncthreads();
// 
//     // Reduce the totals of each input segment. The spine is WARP_SIZE 
//     // threads wide.
//     if(tid < NumSections) {
//         x = storage.shared[tid];
// #pragma unroll
//         for(int offset = 1; offset < NumSections; offset *= 2)
//             x = shfl_add(x, offset, NumSections);
//         storage.shared[tid] = x;
//     }
//     __syncthreads();
// 
//     int reduction = storage.shared[NumSections - 1];
//     __syncthreads();
// 
//     return reduction;
// }

__device__ double __shfl_down64(double var, unsigned int srcLane, int width=32) 
{
        int2 a = *reinterpret_cast<int2*>(&var);
        a.x = __shfl_down(a.x, srcLane, width);
        a.y = __shfl_down(a.y, srcLane, width);
        return *reinterpret_cast<double*>(&a);
}

__inline__ __device__ IndexedSAHSplit warpReduceSum(IndexedSAHSplit val) 
{
    double d; IndexedSAHSplit _v;
#pragma unroll
    for(int offset = warpSize/2; offset > 0; offset /= 2)
    {
        d = __shfl_down64(*(( double * ) &( val.sah)), offset);
        memcpy(&_v, &d, sizeof(double));
        val = ris(_v, val);
    }
    return val;
}

template<int warps>
__inline__ __device__  IndexedSAHSplit blockReduceSum(IndexedSAHSplit val, IndexedSAHSplit neutral) 
{
    __shared__ IndexedSAHSplit shared[warps]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if(lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : neutral;

    if(wid == 0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

template <CTuint blockSize>
__global__ void segReduce(IndexedSAHSplit* splits, CTuint N, CTuint eventCount)
{
    //RETURN_IF_OOB(N);

    CTuint segOffset;
    CTuint segLength;

    segOffset = 2 * g_nodes.contentStart[blockIdx.x];
    segLength= 2 * g_nodes.contentCount[blockIdx.x];

#ifndef USE_OPT_RED
    __shared__ IndexedSAHSplit sdata[blockSize];
     sdata[threadIdx.x].index = 0;
     sdata[threadIdx.x].sah = FLT_MAX;
#endif

    CTuint tid = threadIdx.x;
    CTuint i = tid;

    __shared__ IndexedSAHSplit neutralSplit;
    neutralSplit.index = 0;
    neutralSplit.sah = FLT_MAX;

    IndexedSAHSplit split;
    split.index = 0;
    split.sah = FLT_MAX;

    while(i < segLength) 
    { 
#ifndef USE_OPT_RED
        sdata[tid] = ris(
            sdata[tid], 
            ris(splits[segOffset + i], i + blockSize < segLength ? splits[segOffset + i + blockSize] : neutralSplit)            );
#else
        split /*sdata[tid]*/ = ris(
            split,
            //sdata[tid], 
            ris(splits[segOffset + i], i + blockSize < segLength ? splits[segOffset + i + blockSize] : neutralSplit)            );
#endif
        i += 2 * blockSize;
    }

#ifdef USE_OPT_RED
    split = blockReduceSum<blockSize/32>(split, neutralSplit);
    if(tid == 0) splits[segOffset] = split;
#else

    __syncthreads();

    if(blockSize >= 512) { if(tid < 256) { sdata[tid] = ris(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if(blockSize >= 256) { if(tid < 128) { sdata[tid] = ris(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if(blockSize >= 128) { if(tid <  64) { sdata[tid] = ris(sdata[tid], sdata[tid +  64]); } __syncthreads(); }

    //todo sync weg?
    if(tid < 32) 
    {
        if(blockSize >= 64) sdata[tid] = ris(sdata[tid], sdata[tid + 32]);
        __syncthreads();

        if (blockSize >= 32) sdata[tid] = ris(sdata[tid], sdata[tid + 16]);
        __syncthreads();

        if(blockSize >= 16) sdata[tid] = ris(sdata[tid], sdata[tid + 8]);
        __syncthreads();

        if(blockSize >= 8) sdata[tid] = ris(sdata[tid], sdata[tid + 4]);
        __syncthreads();
        
        if(blockSize >= 4) sdata[tid] = ris(sdata[tid], sdata[tid + 2]);
        __syncthreads();

        if(blockSize >= 2) sdata[tid] = ris(sdata[tid], sdata[tid + 1]);
        __syncthreads();

        //IndexedSAHSplit s = warpReduceSum(sdata[tid]);

        if(tid == 0) splits[segOffset] = sdata[0];
    }
#endif
}

template <CTuint blockSize>
void cudaSegReduce(IndexedSAHSplit* splits, CTuint N, CTuint eventCount, CTuint grid, CTuint block, cudaStream_t stream)
{
    segReduce<blockSize><<<grid, block, 0, stream>>>(splits, N, eventCount);
}

__device__ __forceinline__  int laneid(void)
{
    return threadIdx.x & (warpSize-1);
}

__device__ int warpShflScan(int x)
{
    #pragma unroll
    for(int offset = 1 ; offset < 32 ; offset <<= 1)
    {
        float y = __shfl_up(x, offset);
        if(laneid() >= offset)
        {
            x += y;
        }
    }
    return x;
}

__device__ uint warp_scan(int val, volatile uint* sdata)
{
    uint idx = 2 * threadIdx.x - (threadIdx.x & (warpSize-1));
    sdata[idx] = 0;
    idx += warpSize;
    uint t = sdata[idx] = val;
    sdata[idx] = t = t + sdata[idx-1];
    sdata[idx] = t = t + sdata[idx-2];
    sdata[idx] = t = t + sdata[idx-4];
    sdata[idx] = t = t + sdata[idx-8];
    sdata[idx] = t = t + sdata[idx-16];
   
    return sdata[idx-1];
}

__device__ unsigned int lanemasklt()
{
    const unsigned int lane = threadIdx.x & (warpSize-1);
    return (1<<(lane)) - 1;
}

__device__ unsigned int warpprefixsums(bool p)
{
    const unsigned int mask = lanemasklt();
    unsigned int b = __ballot(p);
    return __popc(b & mask);
}

__device__ CTuint blockPrefixSums(CTuint* sdata, int x)
{
    int warpPrefix = warpShflScan(x);

    int idx = threadIdx.x;
    int warpIdx = idx / warpSize;
    int laneIdx = idx & (warpSize-1); 

    if(laneIdx == warpSize-1) 
    {
        sdata[warpIdx] = warpPrefix;// + x; 
    }

    __syncthreads(); 

    if(idx < warpSize)
    {
        sdata[idx] = warp_scan(sdata[idx], sdata); 
    }

    __syncthreads();

    return sdata[warpIdx] + warpPrefix - x;
}

 __device__ CTuint blockBinaryPrefixSums(CTuint* sdata, int x) 
 { 
     int warpPrefix = warpprefixsums(x);
     int idx = threadIdx.x;
     int warpIdx = idx / warpSize;
     int laneIdx = idx & (warpSize-1); 
     
     if(laneIdx == warpSize-1) 
     {
         sdata[warpIdx] = warpPrefix + x; 
     }

     __syncthreads(); 
     
     if(idx < warpSize)
     {
         sdata[idx] = warp_scan(sdata[idx], sdata); 
     }

     __syncthreads();

     return sdata[warpIdx] + warpPrefix;
 }


template <
    CTuint blockSize, 
    typename Operator, 
    typename T
>
__global__ void completeScan(const T* __restrict g_data, CTuint* scanned, Operator op, CTuint N)
{
    __shared__ uint shrdMem[blockSize];

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

template <
    CTuint blockSize, 
    typename Operator, 
    typename T
>
void cudaCompleteScan(const T* __restrict g_data, CTuint* scanned, Operator op, CTuint N, cudaStream_t stream)
{
    completeScan<blockSize><<<1, blockSize, 0, stream>>>(g_data, scanned, op, N);
}

template <
    CTuint blockSize, 
    CTuint numLines, 
    typename Operator, 
    typename ConstTuple, 
    typename Tuple
>
__global__ void completeScan2NoOpt(ConstTuple g_data, Tuple scanned, Operator op, CTuint N)
{
    __shared__ uint shrdMem[blockSize];

    __shared__ CTuint prefixSum;

    if(threadIdx.x == 0)
    {
        prefixSum = 0;
    }

    CTuint elem = op.GetNeutral();

    if(threadIdx.x < N)
    {
        elem = op(g_data.ts[blockIdx.x][threadIdx.x]);
    }

    CTuint nextElem = op.GetNeutral();

    for(CTuint offset = 0; offset < N; offset += blockSize)
    {
        uint gpos = offset + threadIdx.x;

        if(gpos + blockSize < N)
        {
            nextElem = op(g_data.ts[blockIdx.x][gpos + blockSize]);
        }

        CTuint sum = blockScan<blockSize>(shrdMem, elem);
        //CTuint sum = blockPrefixSums(shrdMem, elem);
        if(gpos < N)
        {
            scanned.ts[blockIdx.x][gpos] = sum + prefixSum - elem;
        }

        __syncthreads();

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum;
        }

        elem = op(nextElem);
    }
}

 template <
    CTuint blockSize, 
    CTuint numLines, 
    typename Operator, 
    typename ConstTuple, 
    typename Tuple
>
__global__ void completeScan2(ConstTuple g_data, Tuple scanned, Operator op, CTuint N)
{
    __shared__ uint shrdMem[blockSize];
    
    __shared__ CTuint prefixSum;

    if(threadIdx.x == 0)
    {
        prefixSum = 0;
    }

    CTuint elem = op.GetNeutral();

    if(threadIdx.x < N)
    {
        elem = op(g_data.ts[blockIdx.x][threadIdx.x]);
    }

    CTuint nextElem = op.GetNeutral();

    for(CTuint offset = 0; offset < N; offset += blockSize)
    {
        uint gpos = offset + threadIdx.x;

        if(gpos + blockSize < N)
        {
            nextElem = op(g_data.ts[blockIdx.x][gpos + blockSize]);
        }

        CTuint sum = blockScan<blockSize>(shrdMem, elem);
        //CTuint sum = blockPrefixSums(shrdMem, elem);
        if(gpos < N)
        {
            scanned.ts[blockIdx.x][gpos] = sum + prefixSum - elem;
        }

        __syncthreads();

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum;
        }

        elem = op(nextElem);
    }
}

// template <
//     CTuint blockSize, 
//     CTuint numLines, 
//     CTuint stepsPerThread,
//     typename Operator, 
//     typename ConstTuple, 
//     typename Tuple
// >
// __global__ void completeScan(ConstTuple g_data, Tuple scanned, Operator op, CTuint N)
// {
//     __shared__ uint shrdMem[blockSize];
//     uint stepMem[stepsPerThread];
//     
//     __shared__ CTuint prefixSum;
//     prefixSum = 0;
// 
//     CTuint step = 0;
// 
// #pragma unroll
//     for(CTuint offset = 0; step < stepsPerThread; offset += blockSize, ++step)
//     {
//         CTuint elem = op(op.GetNeutral());
//         uint gpos = offset + threadIdx.x;
//         if(gpos < N)
//         {
//            elem = op(g_data.ts[blockIdx.x][gpos]);
//         }
//         stepMem[step] = elem;
//     }
// 
//     //__syncthreads();
// 
//     step = 0;
// 
// #pragma unroll
//     for(CTuint offset = 0; step < stepsPerThread; offset += blockSize, ++step)
//     {
//         uint gpos = offset + threadIdx.x;
//         /*
//         CTuint elem = op(op.GetNeutral());
//         if(gpos < N)
//         {
//             elem = op(g_data.ts[blockIdx.x][gpos]);
//         }
//         */
// 
//         CTuint elem = stepMem[step];
// 
//         CTuint sum = blockScan<blockSize>(shrdMem, elem);
//         //CTuint sum = blockPrefixSums(shrdMem, elem);
//         if(gpos < N)
//         {
//             scanned.ts[blockIdx.x][gpos] =sum + prefixSum - elem;
//         }
// 
//         if(threadIdx.x == blockDim.x-1)
//         {
//             prefixSum += sum;
//         }
//         __syncthreads();
//     }
// }

template <
    CTuint blockSize, 
    typename Operator, 
    CTuint numLines, 
    typename ConstTuple, 
    typename Tuple
>
__global__ void segScanN(ConstTuple g_data, Tuple scanned, Tuple sums, Operator op, CTuint N, CTuint offset = 0)
{
    __shared__ uint shrdMem[blockSize];
    uint gpos = blockSize * offset + blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
    for(CTuint i = 0; i < numLines; ++i)
    {
        CTuint elem = op(op.GetNeutral());

        if(gpos < N)
        {
            elem = op(g_data.ts[i][gpos]);
        }

        CTuint sum = blockPrefixSums(shrdMem, elem);

        if(gpos < N)
        {
            scanned.ts[i][gpos] = sum;
        }

        if(threadIdx.x == blockDim.x-1)
        {
            sums.ts[i][offset + blockIdx.x] = sum + elem;
        }
        __syncthreads();
    }
}

template <
    typename Tuple
>
__global__ void spreadScannedSums(Tuple scanned, Tuple prefixSum, uint length)
{
    uint thid = threadIdx.x;
    uint grpId = blockIdx.x;
    uint N = blockDim.x;
    uint gid = N + grpId * N + thid;

    if(gid >= length)
    {
        return;
    }

#pragma unroll
    for(int i = 0; i < 3; ++i)
    {
        scanned.ts[i][gid] += prefixSum.ts[i][grpId+1];
    }
}

template <
    typename TupleS,
    typename TupleSS
>
__global__ void spreadScannedSums4t(TupleS scanned, TupleSS prefixSum, uint length, uint scanSize)
{
    uint tileSize = scanSize;
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= length)
    {
        return;
    }

#pragma unroll
    for(int i = 0; i < 3; ++i)
    {
        uint ps = prefixSum.ts[i][id/tileSize + 1];
        scanned.ts[i][tileSize + id] += ps;
    }
}

template <
    typename Tuple
>
__global__ void spreadScannedSums2(Tuple scanned, Tuple prefixSum, uint length)
{
    const uint elems = 2;
    uint tileSize = 8 * 256 / elems;
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= length)
    {
        return;
    }

#pragma unroll
    for(int i = 0; i < 3; ++i)
    {
        ((uint2*)scanned.ts[i])[tileSize + id].x += prefixSum.ts[i][(elems*id+0)/(elems*tileSize) + 1];
        ((uint2*)scanned.ts[i])[tileSize + id].y += prefixSum.ts[i][(elems*id+1)/(elems*tileSize) + 1];
    }
}

__global__ void spreadScannedSumsSingle(CTuint* scanned, const CTuint* __restrict prefixSum, uint length)
{
    uint thid = threadIdx.x;
    uint grpId = blockIdx.x;
    uint N = blockDim.x;
    uint gid = N + grpId * N + thid;

    if(gid >= length)
    {
        return;
    }

    __shared__ CTuint blockSums;

    if(threadIdx.x == 0)
    {
        blockSums = prefixSum[grpId+1];
    }

    __syncthreads();

    scanned[gid] = scanned[gid] + blockSums;
}

void cudaSpreadScannedSumsSingle(CTuint* scanned, const CTuint* __restrict prefixSum, uint length, CTuint grid, CTuint block, cudaStream_t stream)
{
    spreadScannedSumsSingle<<<grid, block, 0, stream>>>(
        scanned, prefixSum, length);
}

template <
    CTuint blockSize, 
    typename ConstTuple,
    typename Tuple,
    typename Operator
>
__global__ void tripleGroupScan(const ConstTuple g_data, Tuple scanned, Tuple sums, Operator op, CTuint N)
{
    __shared__ CTuint shrdMem[blockSize];

    uint gpos = blockDim.x * blockIdx.x + threadIdx.x;

    CTuint cache[3];
    cache[0] = op.GetNeutral();
    cache[1] = op.GetNeutral();
    cache[2] = op.GetNeutral();

    if(gpos < N)
    {
#pragma unroll
        for(CTbyte i = 0; i < 3; ++i)
        {
            cache[i] = g_data.ts[i][gpos];
        }
    }

#pragma unroll
    for(CTbyte i = 0; i < 3; ++i)
    {
        CTuint elem = op(cache[i]);

        CTuint sum = blockScan<blockSize>(shrdMem, elem);
        if(gpos < N)
        {
            scanned.ts[i][gpos] = sum - elem;
        }

        if(threadIdx.x == blockDim.x-1)
        {
            sums.ts[i][blockIdx.x] = sum;
        }
    }
}

template <
    CTuint blockSize, 
    typename ConstTuple,
    typename Tuple,
    typename Operator
>
__global__ void binaryTripleGroupScan4(const ConstTuple g_data, Tuple scanned, Tuple sums, Operator op, CTuint N)
{
    __shared__ uint shrdMem[blockSize/32];

    const uint LOCAL_SCAN_SIZE = 2 * blockSize;

#pragma unroll
    for(CTbyte i = 0; i < 3; ++i)
    {
        __shared__ uint shrdMem[blockSize/32];

        int globalOffset = blockIdx.x * LOCAL_SCAN_SIZE;

        if(globalOffset >= N)
        {
            return;
        }

        int ai = threadIdx.x;
        int bi = threadIdx.x + (LOCAL_SCAN_SIZE / 2);

        uchar4 a = {0,0,0,0};
        uchar4 b = {0,0,0,0};

        if(4 * (globalOffset + ai) + 3 < N)
        {
            a = ((uchar4*)g_data.ts[i])[globalOffset + ai];
        }    
        else
        {
            int elemsLeft = N - 4 * (globalOffset + ai); 
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                memcpy((void*)&a, (void*)(((uchar4*)g_data.ts[i]) + globalOffset + ai), elemsLeft);
            }
        }

        if(4 * (globalOffset + bi) + 3 < N)
        {
            b = ((uchar4*)g_data.ts[i])[globalOffset + bi];
        }
        else
        {
            int elemsLeft = N - 4 * (globalOffset + bi);
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                memcpy((void*)&b, (void*)(((uchar4*)g_data.ts[i]) + globalOffset + bi), elemsLeft);
            }
        }

        a.x = op(a.x);
        a.y = op(a.y);
        a.z = op(a.z);
        a.w = op(a.w);

        b.x = op(b.x);
        b.y = op(b.y);
        b.z = op(b.z);
        b.w = op(b.w);

        uint partSumA1 = a.y + a.x;
        uint partSumA2 = a.z + partSumA1;

        uint sum0 = __blockScan<blockSize>(shrdMem, partSumA2 + a.w) - (partSumA2 + a.w);

        uint leftSideSum = shrdMem[blockSize/32-1];

        uint partSumB1 = b.y +  b.x;
        uint partSumB2 = b.z + partSumB1;

        __syncthreads();

        uint sum1 = leftSideSum + __blockScan<blockSize>(shrdMem, partSumB2 + b.w) - (partSumB2 + b.w);

        if(4 * (globalOffset + ai) + 3 < N)
        {
            ((uint4*)scanned.ts[i])[globalOffset + ai].x = sum0;
            ((uint4*)scanned.ts[i])[globalOffset + ai].y = sum0 + a.x;
            ((uint4*)scanned.ts[i])[globalOffset + ai].z = sum0 + partSumA1;
            ((uint4*)scanned.ts[i])[globalOffset + ai].w = sum0 + partSumA2;
        }
        else
        {
            int elemsLeft = N - 4 * (globalOffset + ai); 
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                uint3 _a;
                _a.x = sum0;
                _a.y = sum0 + a.x;
                _a.z = sum0 + partSumA1;
                memcpy((void*)(((uint4*)scanned.ts[i]) + globalOffset + ai), (void*)&_a, 4 *elemsLeft);
            }
        }

        if(4 * (globalOffset + bi) + 3 < N)
        {
            ((uint4*)scanned.ts[i])[globalOffset + bi].x = sum1;
            ((uint4*)scanned.ts[i])[globalOffset + bi].y = sum1 + b.x;
            ((uint4*)scanned.ts[i])[globalOffset + bi].z = sum1 + partSumB1;
            ((uint4*)scanned.ts[i])[globalOffset + bi].w = sum1 + partSumB2;
        }
        else
        {
            int elemsLeft = N - 4 * (globalOffset + bi); 
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                uint3 _a;
                _a.x = sum1;
                _a.y = sum1 + b.x;
                _a.z = sum1 + partSumB1;
                memcpy((void*)(((uint4*)scanned.ts[i]) + globalOffset + bi), (void*)&_a, 4 * elemsLeft);
            }
        }

        if(threadIdx.x == blockDim.x-1)
        {
            sums.ts[i][blockIdx.x] = sum1 + partSumB2 + b.w;
        }
    }
}

template <
    CTuint blockSize, 
    typename ConstTuple,
    typename Tuple,
    typename Operator
>
__global__ void binaryTripleGroupScanNoOpt(const ConstTuple g_data, Tuple scanned, Tuple sums, Operator op, CTuint N)
{
    __shared__ CTuint shrdMem[blockSize];

    uint gpos = blockDim.x * blockIdx.x + threadIdx.x;

    CTuint cache[3];
    cache[0] = op.GetNeutral();
    cache[1] = op.GetNeutral();
    cache[2] = op.GetNeutral();

    if(gpos < N)
    {
#pragma unroll
        for(CTbyte i = 0; i < 3; ++i)
        {
            cache[i] = g_data.ts[i][gpos];
        }
    }

#pragma unroll
    for(CTbyte i = 0; i < 3; ++i)
    {
        CTuint elem = op(cache[i]);

        CTuint sum = blockBinaryPrefixSums(shrdMem, elem);
        //CTuint sum = blockScan<blockSize>(shrdMem, elem);
        if(gpos < N)
        {
            scanned.ts[i][gpos] = sum;
        }

        if(threadIdx.x == blockDim.x-1)
        {
            sums.ts[i][blockIdx.x] = sum + elem;
        }
    }
}

template <
    CTuint blockSize, 
    CTuint LOCAL_SCAN_SIZE,
    typename ConstTuple,
    typename TupleS,
    typename TupleSS,
    typename Operator
>
__global__ void binaryTripleGroupScan(const ConstTuple g_data, TupleS scanned, TupleSS sums, Operator op, CTuint N)
{
    __shared__ CTuint shrdMem[blockSize/32];
#pragma unroll
    for(CTbyte i = 0; i < 3; ++i)
    {
       // __shared__ uint shrdMem[blockSize/32];

        uint globalOffset = blockIdx.x * LOCAL_SCAN_SIZE;

        uint gsum = 0;

        uint ai = threadIdx.x;
#pragma unroll
        for(uint offset = 0; offset < LOCAL_SCAN_SIZE; offset += blockSize)
        {
            uchar4 a = {0,0,0,0};

            if(4 * (globalOffset + ai) + 3 < N)
            {
                a = g_data.ts[i][globalOffset + ai];
            }    
            else
            {
                int elemsLeft = N - 4 * (globalOffset + ai); 
                if(elemsLeft > 0 && elemsLeft < 4)
                {
                    memcpy((void*)&a, (void*)(g_data.ts[i] + globalOffset + ai), 4*elemsLeft);
                }
            }

            a.x = op(a.x);
            a.y = op(a.y);
            a.z = op(a.z);
            a.w = op(a.w);

            uint partSumA1 = a.y + a.x;
            uint partSumA2 = a.z + partSumA1;

            uint sum0 = __blockScan<blockSize>(shrdMem, partSumA2 + a.w) - (partSumA2 + a.w) + gsum;

            if(4 * (globalOffset + ai) + 3 < N)
            {
                scanned.ts[i][globalOffset + ai].x = sum0;
                scanned.ts[i][globalOffset + ai].y = sum0 + a.x;
                scanned.ts[i][globalOffset + ai].z = sum0 + partSumA1;
                scanned.ts[i][globalOffset + ai].w = sum0 + partSumA2;
            }
            else
            {
                int elemsLeft = N - 4 * (globalOffset + ai); 
                if(elemsLeft > 0 && elemsLeft < 4)
                {
                    uint3 _a;
                    _a.x = sum0;
                    _a.y = sum0 + a.x;
                    _a.z = sum0 + partSumA1;
                    memcpy((void*)(scanned.ts[i] + globalOffset + ai), (void*)&_a, 4 *elemsLeft);
                }
            }
            
            globalOffset += blockSize;
            gsum += shrdMem[blockSize/32-1];
        }
        if(threadIdx.x == blockDim.x-1)
        {
            sums.ts[i][blockIdx.x] = gsum;
        }
    }
}

template <
    CTuint blockSize, 
    typename Operator,
    typename T
>
__global__ void binaryGroupScan(const T* __restrict g_data, CTuint* scanned, CTuint* sums, Operator op, CTuint N)
{
    __shared__ CTuint shrdMem[blockSize];

    uint gpos = blockDim.x * blockIdx.x + threadIdx.x;

    CTuint elem = op.GetNeutral();
    if(gpos < N)
    {
        elem = op(g_data[gpos]);
    }

    CTuint sum = blockBinaryPrefixSums(shrdMem, elem);

    if(gpos < N)
    {
        scanned[gpos] = sum;
    }

    if(threadIdx.x == blockDim.x-1)
    {
        sums[blockIdx.x] = sum + elem;
    }
}

template <
    CTuint blockSize, 
    typename Operator,
    typename T
>
void cudaBinaryGroupScan(const T* __restrict g_data, CTuint* scanned, CTuint* sums, Operator op, CTuint N, CTuint grid, CTuint block, cudaStream_t stream)
{
    binaryGroupScan<blockSize><<<grid, block, 0, stream>>>(
        g_data, scanned, sums, op, N);
}

#if 1
#if defined DYNAMIC_PARALLELISM

__global__ void dpReduceSAHSplits(IndexedSAHSplit* splits, uint N)
{
    RETURN_IF_OOB(N);

    nutty::DevicePtr<IndexedSAHSplit>::size_type start = 2 * g_nodes.contentStart[id];
    nutty::DevicePtr<IndexedSAHSplit>::size_type length = 2 * g_nodes.contentCount[id];

    IndexedSAHSplit neutralSplit;
    neutralSplit.index = 0;
    neutralSplit.sah = FLT_MAX;
    nutty::DevicePtr<IndexedSAHSplit> ptr_start(splits + start);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    nutty::base::ReduceDP(ptr_start, ptr_start, length, ReduceIndexedSplit(), neutralSplit, stream);
    //cudaDeviceSynchronize();
    //cudaStreamDestroy(stream);
}
#endif
#endif

template <int width, typename Operator>
__global__ void shfl_scan_perblock(const CTbyte* __restrict data, CTuint* scanned, CTuint* g_sums, Operator op, CTuint N)
{
    __shared__ CTuint sums[width];
     CTuint id = ((blockIdx.x * blockDim.x) + threadIdx.x);
//     CTuint lane_id = id % warpSize;
    // determine a warp_id within a block
//    CTuint warp_id = threadIdx.x / warpSize;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = 0;
    int me;
    if(id < N)
    {
        value = (int)op(data[id]);
    }

    me = value;

    value = blockScan<width>(sums, value);

    // Now write out our result
    if(id < N)
    {
        scanned[id] = (CTuint)value - me;
    }

    // last thread has sum, write write out the block's sum
    if(threadIdx.x == blockDim.x-1)
    {
        g_sums[blockIdx.x] = value;
    }
}