#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"
#include <Reduce.h>

#define DYNAMIC_PARALLELISM

#ifdef _DEBUG
#undef DYNAMIC_PARALLELISM
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
__global__ void createEvents3(
    //cuEventLineTriple events,
    const BBox* __restrict primAxisAlignedBB,
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
}

template <CTbyte LONGEST_AXIS>
__global__ void computeSAHSplits3(
    //cuEventLineTriple events, 
    const CTuint* __restrict nodesContentCount,
    const CTuint* __restrict nodesContentStart,
    const BBox* __restrict nodesBBoxes,
    CTuint N,
    CTbyte srcIndex) 
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex;
    BBox bbox;
    CTbyte axis;
    CTreal r = -1;

    EVENT_TRIPLE_HEADER_SRC;

    #pragma unroll
    for(CTbyte i = 0; i < 3; ++i)
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

    CTbyte type = eventsSrc.lines[axis].type[id];

    CTreal split = eventsSrc.lines[axis].indexedEvent[id].v;

    CTuint prefixPrims = nodesContentStart[nodeIndex]; //events.lines[axis].prefixSum[id];

    CTuint primCount = nodesContentCount[nodeIndex];
    
    CTuint startScan = eventsSrc.lines[axis].scannedEventTypeStartMask[id];
    CTuint above = primCount - /*events.lines[axis].scannedEventTypeEndMask[id]*/ (id - startScan) + prefixPrims - type;
    CTuint below = startScan - prefixPrims;

    g_splits.above[id] = above;
    g_splits.below[id] = below;
    g_splits.indexedSplit[id].index = id;

    g_splits.indexedSplit[id].sah = getSAH(bbox, axis, split, below, above);
    g_splits.axis[id] = axis;
    g_splits.v[id] = split;
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

__global__ void setEventsBelongToLeaf(
    const CTbyte* __restrict isLeaf,
    CTbyte* eventIsLeaf,
    CTuint N,
    CTbyte srcIndex)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    eventIsLeaf[id] = isLeaf[eventsSrc.lines[0].nodeIndex[id]] && (eventsSrc.lines[0].type[id] == EVENT_START);
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

__global__ void compactInteriorEventData(
    const CTbyte* __restrict activeNodeIsLeaf,
    const CTuint* __restrict interiorCountScanned,
    const CTuint* __restrict leafContentScanned,
    const CTuint* __restrict leafEventScanned,
    const CTuint* __restrict leafContentCount,
    const CTuint* __restrict leafContentStart, 
    const CTuint* __restrict scannedLeafContent,
    const CTuint* __restrict nodesContentCount,
    const CTbyte* __restrict eventIsLeafMask, 

    CTuint* leafContent, 
    
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

    CTuint offset = 0;
//     if(currentLeafCount)
//     {
//         offset = contentStartOffset;// + scannedLeafContent[oldNodeIndex];// + nodesContentCount[oldNodeIndex];
//     }

    if(!activeNodeIsLeaf[oldNodeIndex])
    {
        CTuint eventsBeforeMe = 2 * leafContentScanned[oldNodeIndex];
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
    //CTuint* leafMask, 
    CTuint* interiorMask,
    CTuint N)
{
    RETURN_IF_OOB(N);
    CTbyte mask = isLeaf[id];
    //leafMask[id] = (mask == 1) * contentCount[id];
    interiorMask[id] = (mask < 2) * (1 ^ mask) * contentCount[id];
}

#if defined DYNAMIC_PARALLELISM

__global__ void dpReduceSAHSplits(Node nodes, IndexedSAHSplit* splits, uint N)
{
    RETURN_IF_OOB(N);

    nutty::DevicePtr<IndexedSAHSplit>::size_type start = 2 * nodes.contentStart[id];
    nutty::DevicePtr<IndexedSAHSplit>::size_type length = 2 * nodes.contentCount[id];

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