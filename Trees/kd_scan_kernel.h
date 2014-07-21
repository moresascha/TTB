#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"
#include <Reduce.h>
#include <device_functions.h>
#include <sm_35_intrinsics.h>

#define DYNAMIC_PARALLELISM

#ifdef _DEBUG
//#undef DYNAMIC_PARALLELISM
#endif


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

/*__constant__ cuEventLineTriple g_eventTripleDst;
__constant__ cuEventLineTriple g_eventTripleSrc; */
__constant__ cuEventLineTriple g_eventTriples[2];

__constant__ Split g_splits;
__constant__ SplitConst g_splitsConst;
__constant__ Node g_nodes;

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
                           
                           const CTuint* __restrict scanVals, CTuint* dst,

                           CTuint N)
{
    RETURN_IF_OOB(N);

    dst[id] = id - scanVals[id];
    leafContentScanned[id] = nodesContentScanned[id] - interiorScannedContent[id];
}

__global__ void reorderEvent3(
    /*cuEventLineTriple dst,
    cuEventLineTriple src,*/
    CTuint N)
{
    RETURN_IF_OOB(N);

    #pragma unroll
    for(CTbyte a = 0; a < 3; ++a)
    {
        CTuint srcIndex = g_eventTriples[0].lines[a].indexedEvent[id].index;
        
        g_eventTriples[1].lines[a].indexedEvent[id] = g_eventTriples[0].lines[a].indexedEvent[id];

        g_eventTriples[1].lines[a].type[id] = g_eventTriples[0].lines[a].type[srcIndex];

        g_eventTriples[1].lines[a].nodeIndex[id] = g_eventTriples[0].lines[a].nodeIndex[srcIndex];

        g_eventTriples[1].lines[a].primId[id] = g_eventTriples[0].lines[a].primId[srcIndex];

        g_eventTriples[1].lines[a].ranges[id] = g_eventTriples[0].lines[a].ranges[srcIndex];
    }
}

template<CTbyte useFOR, CTbyte axis>
__global__ void createEventsAndInit3(
    const BBox* __restrict primAxisAlignedBB,
    const BBox* __restrict sceneBBox,

    CTuint* activeNodesIsLeaf,
    CTuint* nodeToLeafIndex,
    CTbyte* nodeIsLeaf,
    CTuint* nodesContentCount,
    BBox* nodesBBox,

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
        for(CTbyte a = 0; a < 3; ++a)
        {
            g_eventTriples[0].lines[a].type[start_index] = EVENT_START;
            g_eventTriples[0].lines[a].type[end_index] = EDGE_END;

            g_eventTriples[0].lines[a].nodeIndex[start_index] = 0;
            g_eventTriples[0].lines[a].nodeIndex[end_index] = 0;

            g_eventTriples[0].lines[a].indexedEvent[start_index].index = start_index;
            g_eventTriples[0].lines[a].indexedEvent[end_index].index = end_index;

            g_eventTriples[0].lines[a].primId[start_index] = primIndex;
            g_eventTriples[0].lines[a].primId[end_index] = primIndex;

            g_eventTriples[0].lines[a].indexedEvent[start_index].v = getAxis(aabb.m_min, a);
            g_eventTriples[0].lines[a].indexedEvent[end_index].v = getAxis(aabb.m_max, a);

            g_eventTriples[0].lines[a].ranges[start_index] = aabb;
            g_eventTriples[0].lines[a].ranges[end_index] = aabb;
        }
    }

    if(threadIdx.x == 0)
    {
        activeNodesIsLeaf[0] = 0;
        nodeToLeafIndex[0] = 0;
        nodesBBox[0] = sceneBBox[0];
        nodesContentCount[0] = N;
        nodeIsLeaf[0] = 0;
    }
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
            cache[i].nodeIndex = -1;
            break;
        }

        offset += gridDim.x * blockDim.x;

        cache[i].nodeIndex = eventsSrc.lines[0].nodeIndex[idx];
    }

    offset = 0;
#pragma unroll
    for(CTuint i = 0; i < elemsPerThread; ++i)
    {
        if(cache[i].nodeIndex != -1)
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
            CTbyte axis = (CTbyte)getLongestAxis(cache[i].bbox.m_max - cache[i].bbox.m_min);

            CTbyte type = eventsSrc.lines[axis].type[idx];

            CTreal split = eventsSrc.lines[axis].indexedEvent[idx].v;

            CTuint endScan = eventsSrc.lines[axis].scannedEventTypeEndMask[idx];
            CTuint startScan = idx - endScan;
            CTuint above = cache[i].primCount - endScan + cache[i].prefixPrims - type;
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

__device__ __forceinline bool isIn(BBox& bbox, CTbyte axis, CTreal v)
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

    CTbyte axis = id / eventsPerAxisN;

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
            CTbyte splitAxis = g_splitsConst.axis[isplit.index];
            CTreal split = g_splitsConst.v[isplit.index];
            BBox bbox = src.lines[axis].ranges[i];
            CTreal v = src.lines[axis].indexedEvent[i].v;    
            CTbyte type = src.lines[axis].type[i];

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
    CTbyte myAxis,
    CTuint eventCount)
{
    RETURN_IF_OOB(eventCount);

    CTuint nodeIndex = src.nodeIndex[id];
    CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];

    IndexedSAHSplit isplit = g_splitsConst.indexedSplit[eventsLeftFromMe];
    
    CTbyte splitAxis = g_splitsConst.axis[isplit.index];

    BBox bbox = src.ranges[id];

    CTreal v = src.indexedEvent[id].v;
    CTreal split = g_splitsConst.v[isplit.index];
    CTbyte type = src.type[id];

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
        dst.nodeIndex[N + id] = 2 * nodeIndex+ 1;
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
    CTbyte splitAxis = g_splitsConst.axis[isplit.index];
    CTreal split = g_splitsConst.v[isplit.index];
    CTuint N = eventsLeftFromMe + 2 * nodeContentCount[nodeIndex];

    #pragma unroll
    for(CTbyte axis = 0; axis < 3; ++axis)
    {
        BBox bbox = eventsSrc.lines[axis].ranges[id];
        CTreal v = eventsSrc.lines[axis].indexedEvent[id].v;    
        CTbyte type = eventsSrc.lines[axis].type[id];

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

// __global__ void clipEventsLinearOnAxis(
//     cuEventLineTriple dst,
//     cuEventLineTriple src,
//     const CTuint* __restrict nodeContentStart,
//     const CTuint* __restrict nodeContentCount,
//     Split splits,
//     CTuint count)
// {
//     RETURN_IF_OOB(3 * count);
// 
//     CTbyte axis = id / count;
//     id = id % count;
// 
//     CTuint nodeIndex = src.getLine(0).nodeIndex[id];
//     CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];
//     IndexedSAHSplit isplit = splits.indexedSplit[eventsLeftFromMe];
//     CTbyte splitAxis = splits.axis[isplit.index];
//     CTreal split = splits.v[isplit.index];
//     CTuint N = eventsLeftFromMe + 2 * nodeContentCount[nodeIndex];
// 
//     //for(CTbyte axis = 0; axis < 3; ++axis)
//     {
//         BBox bbox = src.getLine(axis).ranges[id];
//         CTreal v = src.getLine(axis).indexedEvent[id].v;    
//         CTbyte type = src.getLine(axis).type[id];
// 
//         if(getAxis(bbox.m_min, splitAxis) <= split && getAxis(bbox.m_max, splitAxis) <= split)
//         {
//             dst.getLine(axis).mask[eventsLeftFromMe + id] = 1;
// 
//             //left
//             copyEvent(dst.getLine(axis), src.getLine(axis), eventsLeftFromMe + id, id);
//             dst.getLine(axis).nodeIndex[eventsLeftFromMe + id] = 2 * nodeIndex;
//         }
//         else if(getAxis(bbox.m_min, splitAxis) >= split && getAxis(bbox.m_max, splitAxis) >= split)
//         {
//             dst.getLine(axis).mask[N + id] = 1;
// 
//             //right
//             copyEvent(dst.getLine(axis), src.getLine(axis), N + id, id);
//             dst.getLine(axis).nodeIndex[N + id] = 2 * nodeIndex + 1;
//         }
//         else
//         {
//             dst.getLine(axis).mask[eventsLeftFromMe + id] = 1;
//             dst.getLine(axis).mask[N + id] = 1;
// 
//             //both
//             clipCopyEvent(dst.getLine(axis), src.getLine(axis), eventsLeftFromMe + id, id);
//             CTreal save = getAxis(bbox.m_max, splitAxis);
//             setAxis(bbox.m_max, splitAxis, split);
//             dst.getLine(axis).nodeIndex[eventsLeftFromMe + id] = 2 * nodeIndex;
//             dst.getLine(axis).ranges[eventsLeftFromMe + id] = bbox;
// 
//             setAxis(bbox.m_max, splitAxis, save);
//             clipCopyEvent(dst.getLine(axis), src.getLine(axis), N + id, id);
//             setAxis(bbox.m_min, splitAxis, split);
//             dst.getLine(axis).nodeIndex[N + id] = 2 * nodeIndex+ 1;
//             dst.getLine(axis).ranges[N + id] = bbox;
// 
//             if(axis == splitAxis)
//             {
//                 CTuint right = !(split > v || (v == split && type == EDGE_END));
//                 if(right)
//                 {
//                     dst.getLine(axis).indexedEvent[eventsLeftFromMe + id].v = split;
//                 }
//                 else
//                 {
//                     dst.getLine(axis).indexedEvent[N + id].v = split;
//                 }
//             }
//         }
//     }
// }

// __global__ void compactEventLine(
//     cuEventLine dst,
//     cuEventLine src,
//     const CTuint* __restrict mask,
//     const CTuint* __restrict prefixSum,
//     CTuint N)
// {
//     RETURN_IF_OOB(N);
// 
//     if(mask[id])
//     {
//         copyEvent(dst, src, prefixSum[id], id);
//     }
// }

__global__ void clearMasks(CTbyte* a, CTbyte* b, CTbyte* c, CTuint N)
{
    RETURN_IF_OOB(N);
    a[id] = 0;
    b[id] = 0;
    c[id] = 0;
}

__global__ void clearMasks3(CTbyte3* a, CTuint N)
{
    RETURN_IF_OOB(N);
    CTbyte3 v = {0};
    a[id] = v;
}

__global__ void createClipMask(
    cuClipMaskArray clipMask,
    CTbyte* mask3,
    const CTuint* __restrict nodeContentStart,
    const CTuint* __restrict nodeContentCount,
    CTuint count,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(count);
    EVENT_TRIPLE_HEADER_SRC;

    CTuint nodeIndex = eventsSrc.lines[0].nodeIndex[id];
    CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];
    IndexedSAHSplit isplit = g_splitsConst.indexedSplit[eventsLeftFromMe];
    CTbyte splitAxis = g_splitsConst.axis[isplit.index];
    CTreal split = g_splitsConst.v[isplit.index];
    CTuint N = eventsLeftFromMe + 2 * nodeContentCount[nodeIndex];
    
#pragma unroll
    for(CTbyte axis = 0; axis < 3; ++axis)
    {
        const cuEventLine& __restrict srcLine = eventsSrc.lines[axis];

        BBox& bbox = srcLine.ranges[id];
        CTreal v = srcLine.indexedEvent[id].v;    
        CTbyte type = srcLine.type[id];

        CTreal minAxis = getAxis(bbox.m_min, splitAxis);
        CTreal maxAxis = getAxis(bbox.m_max, splitAxis);
        cuClipMask& cm = clipMask.mask[axis];
        CTbyte ma = 0;
        if(minAxis <= split && maxAxis <= split)
        {
            //left
            setLeft(ma);
            setAxis(ma, splitAxis);

            //cm.mask[eventsLeftFromMe + id] = ma;
            mask3[3 * (eventsLeftFromMe + id) + axis] = ma;
            cm.index[eventsLeftFromMe + id] = id;
        }
        else if(minAxis >= split && maxAxis >= split)
        {
            //right
            setRight(ma);
            setAxis(ma, splitAxis);
            cm.index[N + id] = id;
            //cm.mask[N + id] = ma;
            mask3[3 * (N + id) + axis] = ma;
        }
        else
        {
            cm.index[N + id] = id;
            cm.newSplit[N + id] = split;
            cm.newSplit[eventsLeftFromMe + id] = split;
            cm.index[eventsLeftFromMe + id] = id;

            CTbyte mar = 0;
            //both
            setLeft(ma);
            setAxis(ma, splitAxis);
            
            setRight(mar);
            setAxis(mar, splitAxis);

            setOLappin(ma);
            setOLappin(mar);
   
            if(axis == splitAxis)
            {
                CTuint right = !(split > v || (v == split && type == EDGE_END));
                if(right)
                {
                    ma |= 0x40;
                }
                else
                {
                    mar |= 0x40;
                }
            }
            mask3[3 * (eventsLeftFromMe + id) + axis] = ma;
            mask3[3 * (N + id) + axis] = mar;
            //cm.mask[eventsLeftFromMe + id] = ma;
            //cm.mask[N + id] = mar;
        }
    }
}

struct Sums
{
    const CTuint* __restrict prefixSum[3];
};

__global__ void compactEventLineV2(
    cuClipMaskArray clipArray,
    const CTbyte* __restrict mask3,
    //Sums prefixSum,
    const CTuint* __restrict prefixSum,
    CTuint N,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    #pragma unroll
    for(CTbyte i = 0; i < 3; ++i)
    {
        CTbyte mask = /*clipArray.mask[i].mask[id]; */ mask3[3 * id + i]; 
        CTuint eventIndex = clipArray.mask[i].index[id];
        const cuEventLine& __restrict srcLine = eventsSrc.lines[i];
        cuEventLine& dstLine = eventsDst.lines[i];
        //const CTuint* __restrict pfs = prefixSum.prefixSum[i];

        if(isSet(mask))
        {
            CTuint dstAdd = prefixSum[3 * id + i]; //pfs[id];
            CTbyte splitAxis = getAxisFromMask(mask);
            bool right = isRight(mask);

            if(i == 0)
            {
                CTuint nnodeIndex = 2 * srcLine.nodeIndex[eventIndex] + (CTuint)right;
                dstLine.nodeIndex[dstAdd] = nnodeIndex; 
            }

            IndexedEvent e = srcLine.indexedEvent[eventIndex];
            BBox bbox = srcLine.ranges[eventIndex];

            if(isOLappin(mask))
            {
                CTreal split = clipArray.mask[i].newSplit[id];
                setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
                if(i == splitAxis && ((mask & 0x40) == 0x40))
                {
                    e.v = split;
                }
            }

            dstLine.indexedEvent[dstAdd] = e;
            dstLine.primId[dstAdd] = srcLine.primId[eventIndex];
            dstLine.ranges[dstAdd] = bbox;
            dstLine.type[dstAdd] = srcLine.type[eventIndex];
        }
    }
}

__global__ void compactEventLine3(
    Sums prefixSum,
    CTuint N,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;
    #pragma unroll
    for(CTbyte i = 0; i < 3; ++i)
    {
        if(eventsSrc.lines[i].mask[id])
        {
            CTuint dstAdd = prefixSum.prefixSum[i][id];
            if(i == 0)
            {
                eventsDst.lines[i].nodeIndex[dstAdd] = eventsSrc.lines[i].nodeIndex[id];
            }

            eventsDst.lines[i].indexedEvent[dstAdd] = eventsSrc.lines[i].indexedEvent[id];
            eventsDst.lines[i].primId[dstAdd] = eventsSrc.lines[i].primId[id];
            eventsDst.lines[i].ranges[dstAdd] = eventsSrc.lines[i].ranges[id];
            eventsDst.lines[i].type[dstAdd] = eventsSrc.lines[i].type[id];
        }
    } 

/*
    CTuint nodeIndex;
    IndexedEvent event;
    CTuint primId;
    BBox range;
    CTbyte type;
    CTuint dstAdd;

    CTuint masks[3];
    masks[0] = src.getLine(0).mask[id];
    masks[1] = src.getLine(1).mask[id];
    masks[2] = src.getLine(2).mask[id];

    if(masks[0])
    {
        dstAdd = prefixSum.prefixSum[0][id];
        nodeIndex = src.getLine(0).nodeIndex[id];
        event = src.getLine(0).indexedEvent[id];
        primId = src.getLine(0).primId[id];
        range = src.getLine(0).ranges[id];
        type = src.getLine(0).type[id];

        dst.getLine(0).nodeIndex[dstAdd] = nodeIndex;
        dst.getLine(0).indexedEvent[dstAdd] = event;
        dst.getLine(0).primId[dstAdd] = primId;
        dst.getLine(0).ranges[dstAdd] = range;
        dst.getLine(0).type[dstAdd] = type;
    }

    if(masks[1])
    {
        dstAdd = prefixSum.prefixSum[1][id];
        event = src.getLine(1).indexedEvent[id];
        primId = src.getLine(1).primId[id];
        range = src.getLine(1).ranges[id];
        type = src.getLine(1).type[id];

        dst.getLine(1).indexedEvent[dstAdd] = event;
        dst.getLine(1).primId[dstAdd] = primId;
        dst.getLine(1).ranges[dstAdd] = range;
        dst.getLine(1).type[dstAdd] = type;
    }

    if(masks[2])
    {
        dstAdd = prefixSum.prefixSum[2][id];
        event = src.getLine(2).indexedEvent[id];
        primId = src.getLine(2).primId[id];
        range = src.getLine(2).ranges[id];
        type = src.getLine(2).type[id];

        dst.getLine(2).indexedEvent[dstAdd] = event;
        dst.getLine(2).primId[dstAdd] = primId;
        dst.getLine(2).ranges[dstAdd] = range;
        dst.getLine(2).type[dstAdd] = type;
    }
    */
}

__global__ void initInteriorNodes(
    const CTuint* __restrict activeNodes,
    const CTuint* __restrict activeNodesThisLevel,
    const BBox* __restrict oldBBoxes,
    BBox* newBBoxes,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* newActiveNodes,
    CTbyte* activeNodesIsLeaf,
    CTuint childOffset,
    CTuint nodeOffset,
    CTuint N,
    CTuint* oldNodeContentStart,
    CTbyte* gotLeaves,
    CTbyte makeLeaves)
{
    RETURN_IF_OOB(N);

    CTuint can = activeNodesThisLevel[id];
    CTuint an = activeNodes[id];
    CTuint edgesBeforeMe = 2 * oldNodeContentStart[can];

    IndexedSAHSplit split = g_splitsConst.indexedSplit[edgesBeforeMe];

    CTbyte axis = g_splitsConst.axis[split.index];
    CTreal s = g_splitsConst.v[split.index];
    CTuint below = g_splitsConst.below[split.index];
    CTuint above = g_splitsConst.above[split.index];

    CTuint nodeId = nodeOffset + an;
    g_nodes.split[nodeId] = s;
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

    splitAABB(&oldBBoxes[id], s, axis, &l, &r);

    if(below > MAX_ELEMENTS_PER_LEAF)
    {
        newBBoxes[2 * dst + 0] = l;
    }

    if(above > MAX_ELEMENTS_PER_LEAF)
    {
        newBBoxes[2 * dst + 1] = r;
    }
}

__global__ void setEventsBelongToLeafAndSetNodeIndex(
    const CTbyte* __restrict isLeaf,
    CTbyte* eventIsLeaf,
    CTuint* nodeToLeafIndex,
    CTuint N,
    CTuint N2,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    eventIsLeaf[id] = isLeaf[eventsSrc.lines[0].nodeIndex[id]] && (eventsSrc.lines[0].type[id] == EVENT_START);

    if(id < N2)
    {
        nodeToLeafIndex[id] = 0;
    }
}

__global__ void compactLeafData(
    CTuint* leafContent,
    const CTuint* __restrict leafContentCount,
    const CTuint* __restrict leafContentStart, 
    const CTbyte* __restrict eventIsLeafMask, 
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

__global__ void compactMakeLeavesData(
    const CTbyte* __restrict activeNodeIsLeaf,
    const CTuint* __restrict interiorCountScanned,
    const CTuint* __restrict scannedLeafContent,
    const CTuint* __restrict leafEventScanned,

    const CTuint* __restrict nodesContentCount,
    const CTbyte* __restrict eventIsLeafMask,

    const CTuint* __restrict leafCountScan, 
    const CTuint* __restrict interiorCountScan,

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
    CTuint N)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    CTuint oldNodeIndex = eventsSrc.lines[0].nodeIndex[id];

    if(!activeNodeIsLeaf[oldNodeIndex])
    {
        CTuint eventsBeforeMe = 2 * scannedLeafContent[oldNodeIndex];
        CTuint nodeIndex = interiorCountScanned[oldNodeIndex];

        eventsDst.lines[0].nodeIndex[id - eventsBeforeMe] = nodeIndex;

        #pragma unroll
        for(CTbyte i = 0; i < 3; ++i)
        {
            const cuEventLine& __restrict srcLine = eventsSrc.lines[i];
            cuEventLine& dstLine = eventsDst.lines[i];
            dstLine.indexedEvent[id - eventsBeforeMe] = srcLine.indexedEvent[id];
            dstLine.primId[id - eventsBeforeMe] = srcLine.primId[id];
            dstLine.ranges[id - eventsBeforeMe] = srcLine.ranges[id];
            dstLine.type[id - eventsBeforeMe] = srcLine.type[id];
        }
    }
    else if(eventIsLeafMask[id])
    {
        leafContent[leafContentStartOffset + leafEventScanned[id]] = eventsSrc.lines[0].primId[id];
    }

    if(id < nodeCount)
    {
        CTbyte leafMask = activeNodeIsLeaf[id];
        CTuint dst = leafCountScan[id];
        if(leafMask && leafMask < 2)
        {
            leafContentStart[currentLeafCount + dst] = leafContentStartOffset + scannedLeafContent[id];
            leafContentCount[currentLeafCount + dst] = nodesContentCount[id];
            nodeToLeafIndex[activeNodes[id] + offset] = currentLeafCount + isLeafScan[id];
        }
        else
        {
            CTuint dst = interiorCountScan[id];
            newActiveNodes[dst] = activeNodes[id];
            newContentCount[dst] = nodesContentCount[id];
            newContentStart[dst] = scannedInteriorContent[id];
            newBBxes[dst] = nodeBBoxes[id];
        }
    }
}

__global__ void compactLeafNInteriorNodeData(
    const CTuint* __restrict scannedInteriorContent,
    const CTuint* __restrict scannedLeafContent,
    const CTuint* __restrict nodesContentCount,
    const CTuint* __restrict nodesContentStart,
    const BBox* __restrict nodeBBoxes,
    const CTbyte* __restrict isLeaf, 

    const CTuint* __restrict leafCountScan, 
    const CTuint* __restrict interiorCountScan,
    const CTuint* __restrict activeNodes,

    const CTuint* __restrict isLeafScan,

    CTuint* nodeToLeafIndex,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* leafContentStart, 
    CTuint* leafContentCount,
    CTuint* newActiveNodes,
    BBox* newBBxes,
    CTuint leafCount,
    CTuint contentStartOffset,
    CTuint offset,
    CTuint N)
{
    RETURN_IF_OOB(N);

    //     if(LAST_LEVEL)
    //     {
    //         leafContentStart[lastLeafCount + id] = contentStartOffset + scannedLeafContentstart[id];
    //         leafContentCount[lastLeafCount + id] = nodesContentCount[id];
    //         return;
    //     }
    CTbyte leafMask = isLeaf[id];
    CTuint dst = leafCountScan[id];
    if(leafMask && leafMask < 2)
    {
        leafContentStart[leafCount + dst] = contentStartOffset + scannedLeafContent[id];
        leafContentCount[leafCount + dst] = nodesContentCount[id];
        nodeToLeafIndex[activeNodes[id] + offset] = leafCount + isLeafScan[id];
    }
    else
    {
        CTuint dst = interiorCountScan[id];
        newActiveNodes[dst] = activeNodes[id];
        newContentCount[dst] = nodesContentCount[id];
        newContentStart[dst] = scannedInteriorContent[id]; //scannedInteriorContentstart[id];
        newBBxes[dst] = nodeBBoxes[id];
    }
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

__global__ void makeSeperateScans(cuEventLineTriple events, const CTuint3* __restrict src, CTuint N)
{
    RETURN_IF_OOB(N);
     CTuint3 val = src[id];
     events.lines[0].tmpType[id] = val.x;
     events.lines[0].tmpType[id] = val.y;
     events.lines[0].tmpType[id] = val.z;
}

__global__ void createInteriorContentCountMasks(
    const CTbyte* __restrict isLeaf,
    const CTuint* __restrict contentCount,
    CTuint* interiorMask,
    CTuint N)
{
    RETURN_IF_OOB(N);
    CTbyte mask = isLeaf[id];
    interiorMask[id] = (mask < 2) * (1 ^ mask) * contentCount[id];
}

__device__ IndexedSAHSplit ris(IndexedSAHSplit t0, IndexedSAHSplit t1)
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

template <CTuint blockSize>
__global__ void segReduce(IndexedSAHSplit* splits, CTuint N)
{
    RETURN_IF_OOB(N);

    __shared__ CTuint segOffset;
    __shared__ CTuint segLength;

    segOffset = 2 * g_nodes.contentStart[blockIdx.x];
    segLength= 2 * g_nodes.contentCount[blockIdx.x];

    __shared__ IndexedSAHSplit sdata[blockSize];

    CTuint tid = threadIdx.x;
    CTuint i = tid;

    __shared__ IndexedSAHSplit neutralSplit;
    neutralSplit.index = 0;
    neutralSplit.sah = FLT_MAX;

    sdata[tid].index = 0;
    sdata[tid].sah = FLT_MAX;

    while(i < segLength) 
    { 
        sdata[tid] = ris(
            
            sdata[tid], 
            ris(splits[segOffset + i], i + blockSize < segLength ? splits[segOffset + i + blockSize] : neutralSplit)
            
            );

        i += 2 * blockSize;
    }

    __syncthreads();

    if(blockSize >= 512) { if(tid < 256) { sdata[tid] = ris(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if(blockSize >= 256) { if(tid < 128) { sdata[tid] = ris(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if(blockSize >= 128) { if(tid <  64)  { sdata[tid] = ris(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

    //todo sync weg?
    if(tid < 32) 
    {
        if (blockSize >= 64) sdata[tid] = ris(sdata[tid], sdata[tid + 32]);
        __syncthreads();

        if (blockSize >= 32) sdata[tid] = ris(sdata[tid], sdata[tid + 16]);
        __syncthreads();

        if (blockSize >= 16) sdata[tid] = ris(sdata[tid], sdata[tid + 8]);
        __syncthreads();

        if (blockSize >= 8) sdata[tid] = ris(sdata[tid], sdata[tid + 4]);
        __syncthreads();

        if (blockSize >= 4) sdata[tid] = ris(sdata[tid], sdata[tid + 2]);
        __syncthreads();

        if (blockSize >= 2) sdata[tid] = ris(sdata[tid], sdata[tid + 1]);
        //__syncthreads();
    }

    if(tid == 0) splits[segOffset] = sdata[0];
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

 template <
     CTuint count, 
     typename T
 >
 struct Tuple
 {
     T* ts[count];
 };

 template <
     CTuint count, 
     typename T
 >
 struct ConstTuple
 {
     const T* __restrict ts[count];
 };

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
    prefixSum = 0;

    for(CTuint offset = 0; offset < N; offset += blockSize)
    {
        uint gpos = offset + threadIdx.x;
        CTuint elem = op(op.GetNeutral());
        if(gpos < N)
        {
            elem = op(g_data.ts[blockIdx.x][gpos]);
        }

        CTuint sum = blockScan<blockSize>(shrdMem, elem);
        //CTuint sum = blockPrefixSums(shrdMem, elem);
        if(gpos < N)
        {
            scanned.ts[blockIdx.x][gpos] =sum + prefixSum - elem;
        }

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum;
        }
        __syncthreads();
    }
}

template <
    CTuint blockSize, 
    CTuint numLines, 
    CTuint stepsPerThread,
    typename Operator, 
    typename ConstTuple, 
    typename Tuple
>
__global__ void completeScan(ConstTuple g_data, Tuple scanned, Operator op, CTuint N)
{
    __shared__ uint shrdMem[blockSize];
    uint stepMem[stepsPerThread];
    
    __shared__ CTuint prefixSum;
    prefixSum = 0;

    CTuint step = 0;

#pragma unroll
    for(CTuint offset = 0; step < stepsPerThread; offset += blockSize, ++step)
    {
        CTuint elem = op(op.GetNeutral());
        uint gpos = offset + threadIdx.x;
        if(gpos < N)
        {
           elem = op(g_data.ts[blockIdx.x][gpos]);
        }
        stepMem[step] = elem;
    }

    //__syncthreads();

    step = 0;

#pragma unroll
    for(CTuint offset = 0; step < stepsPerThread; offset += blockSize, ++step)
    {
        uint gpos = offset + threadIdx.x;
        /*
        CTuint elem = op(op.GetNeutral());
        if(gpos < N)
        {
            elem = op(g_data.ts[blockIdx.x][gpos]);
        }
        */

        CTuint elem = stepMem[step];

        CTuint sum = blockScan<blockSize>(shrdMem, elem);
        //CTuint sum = blockPrefixSums(shrdMem, elem);
        if(gpos < N)
        {
            scanned.ts[blockIdx.x][gpos] =sum + prefixSum - elem;
        }

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum;
        }
        __syncthreads();
    }
}

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
        CTbyte elem = op(op.GetNeutral());

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

    CTuint cache[3];
    __shared__ CTuint blockSums[3];

#pragma unroll
    for(int i = 0; i < 3; ++i)
    {
        cache[i] = scanned.ts[i][gid];

        if(threadIdx.x == 0)
        {
            blockSums[i] = prefixSum.ts[i][grpId+1];
        }
    }

    __syncthreads();

#pragma unroll
    for(int i = 0; i < 3; ++i)
    {
        scanned.ts[i][gid] = cache[i] + blockSums[i];
    }
}

template <
    CTuint blockSize, 
    typename ConstTuple,
    typename Tuple
>
__global__ void binaryTripleGroupScan(const ConstTuple g_data, Tuple scanned, Tuple sums, CTuint N)
{
    __shared__ CTuint shrdMem[blockSize];

    uint gpos = blockDim.x * blockIdx.x + threadIdx.x;

    CTbyte cache[3];
    cache[0] = 0;
    cache[1] = 0;
    cache[2] = 0;

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
        CTbyte elem = cache[i]; //op(cache[i]);

        CTuint sum = blockBinaryPrefixSums(shrdMem, elem);

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
    CTuint lane_id = id % warpSize;
    // determine a warp_id within a block
    CTuint warp_id = threadIdx.x / warpSize;

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