#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"

#define RETURN_IF_OOB(__N) \
    CTuint id = GlobalId; \
    if(id >= __N) \
    { \
    return; \
    }

__global__ void reorderEvent3(
    cuEventLineTriple src,
    cuEventLineTriple dst,
    CTuint N)
{
    RETURN_IF_OOB(N);

    for(CTbyte a = 0; a < 3; ++a)
    {
        CTuint srcIndex = src.lines[a].indexedEvent[id].index;
        
        dst.lines[a].indexedEvent[id] = src.lines[a].indexedEvent[id];

        dst.lines[a].type[id] = src.lines[a].type[srcIndex];

        dst.lines[a].nodeIndex[id] = src.lines[a].nodeIndex[srcIndex];

        dst.lines[a].primId[id] = src.lines[a].primId[srcIndex];

        dst.lines[a].prefixSum[id] = src.lines[a].prefixSum[srcIndex];

        dst.lines[a].ranges[id] = src.lines[a].ranges[srcIndex];
    }
}

template<CTbyte useFOR, CTbyte axis>
__global__ void createEvents3(
    cuEventLineTriple events,
    const BBox* __restrict primAxisAlignedBB,
    NodeContent nodesContent,
    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint primIndex = id;
    BBox aabb = primAxisAlignedBB[primIndex];

    CTuint start_index = 2 * id + 0;
    CTuint end_index = 2 * id + 1;

    if(useFOR)
    {
        for(CTbyte a = 0; a < 3; ++a)
        {
            events.getLine(a).type[start_index] = EDGE_START;
            events.getLine(a).type[end_index] = EDGE_END;

            events.getLine(a).nodeIndex[start_index] = 0;
            events.getLine(a).nodeIndex[end_index] = 0;

            events.getLine(a).indexedEvent[start_index].index = start_index;
            events.getLine(a).indexedEvent[end_index].index = end_index;

            events.getLine(a).primId[start_index] = primIndex;
            events.getLine(a).primId[end_index] = primIndex;

            events.getLine(a).prefixSum[start_index] = 0;
            events.getLine(a).prefixSum[end_index] = 0;

            events.getLine(a).indexedEvent[start_index].v = getAxis(aabb.m_min, a);
            events.getLine(a).indexedEvent[end_index].v = getAxis(aabb.m_max, a);

            events.getLine(a).ranges[start_index] = aabb;
            events.getLine(a).ranges[end_index] = aabb;
        }
    }
}

template <CTbyte LOGNEST_AXIS>
__global__ void computeSAHSplits3(
    cuEventLineTriple events, 
    const CTuint* __restrict nodesContentCount,
    Split splits,
    const BBox* __restrict nodesBBoxes,
    CTuint N) 
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex;
    BBox bbox;
    CTbyte axis;
    CTreal r = -1;

    for(CTbyte i = 0; i < 3; ++i)
    {
        CTuint ni = events.lines[i].nodeIndex[id];
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

    CTbyte type = events.lines[axis].type[id];

    CTreal split = events.lines[axis].indexedEvent[id].v;

    CTuint prefixPrims = events.lines[axis].prefixSum[id]/2;

    CTuint primCount = nodesContentCount[nodeIndex];
    
    CTuint above = primCount - events.lines[axis].scannedEventTypeEndMask[id] + prefixPrims - type;
    CTuint below = events.lines[axis].scannedEventTypeStartMask[id] - prefixPrims;

    splits.above[id] = above;
    splits.below[id] = below;
    splits.indexedSplit[id].index = id;

    splits.indexedSplit[id].sah = getSAH(bbox, axis, split, below, above);
    splits.axis[id] = axis;
    splits.v[id] = split;
}

__global__ void classifyEvents3(
    Node nodes,
    Event events,
    Split splits,
    const CTbyte* isLeaf,
    CTuint* edgeMask,
    const CTuint* contentStartAdd,
    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = events.nodeIndex[id];

    if(isLeaf[nodeIndex])
    {
        edgeMask[id] = 0;
        events.nodeIndex[id] = -1;
        return;
    }

    CTbyte type = events.type[id];

    CTuint edgesBeforeMe = 2 * contentStartAdd[nodeIndex];
    IndexedSAHSplit is = splits.indexedSplit[edgesBeforeMe];

    CTreal split = splits.v[is.index];

    CTreal v = events.indexedEdge[id].v;

    CTuint right = !(split > v || (v == split && type == EDGE_END));

    if(right)
    {
        edgeMask[id] = type == EDGE_START ? 0 : 1;
    }
    else
    {
        edgeMask[id] = type == EDGE_START ? 1 : 0;
    }

    nodeIndex = 2 * nodeIndex + (right ? 1 : 0);
    events.nodeIndex[id] = nodeIndex;
}

__global__ void setEventBelongsToLeaf(
    NodeContent nodesContent,
    const CTbyte* __restrict isLeaf,
    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = nodesContent.nodeIndex[id];
    nodesContent.primIsLeaf[id] = isLeaf[nodeIndex] > 0;
}

__device__ bool isIn(BBox& bbox, CTbyte axis, CTreal v)
{
    return getAxis(bbox.GetMin(), axis) <= v && v <= getAxis(bbox.GetMax(), axis);
}

__device__ void copyEvent(cuEventLine dst, cuEventLine src, CTuint dstIndex, CTuint srcIndex)
{
    dst.indexedEvent[dstIndex] = src.indexedEvent[srcIndex];
    dst.nodeIndex[dstIndex] = src.nodeIndex[srcIndex];
    dst.prefixSum[dstIndex] = src.prefixSum[srcIndex];
    dst.primId[dstIndex] = src.primId[srcIndex];
    dst.ranges[dstIndex] = src.ranges[srcIndex];
    dst.type[dstIndex] = src.type[srcIndex];
}

__device__ void clipCopyEvent(cuEventLine dst, cuEventLine src, CTuint dstIndex, CTuint srcIndex)
{
    dst.indexedEvent[dstIndex] = src.indexedEvent[srcIndex];
    //dst.nodeIndex[dstIndex] = src.nodeIndex[srcIndex];
    //dst.prefixSum[dstIndex] = src.prefixSum[srcIndex];
    dst.primId[dstIndex] = src.primId[srcIndex];
    //dst.ranges[dstIndex] = src.ranges[srcIndex];
    dst.type[dstIndex] = src.type[srcIndex];
}

__global__ void clipEvents(
    cuEventLine dst,
    cuEventLine src,
    CTuint* mask,
    const CTuint* __restrict nodeContentStart,
    Split splits,
    CTbyte myAxis,
    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint eventsLeftFromMe = 2 * nodeContentStart[src.nodeIndex[id]];

    IndexedSAHSplit isplit = splits.indexedSplit[eventsLeftFromMe];
    
    CTbyte splitAxis = splits.axis[isplit.index];

    BBox bbox = src.ranges[id];

    CTreal v = src.indexedEvent[id].v;
    CTreal split = splits.v[isplit.index];
    CTbyte type = src.type[id];

    //CTuint above = splits.above[isplit.index];
    //CTuint below = splits.below[isplit.index];

//     CTuint right = !(split > v || (v == split && type == EDGE_END));
// 
//     if(right)
//     {
//         edgeMask[id] = type == EDGE_START ? 0 : 1;
//     }
//     else
//     {
//         edgeMask[id] = type == EDGE_START ? 1 : 0;
//     }
// 
//     nodeIndex = 2 * nodeIndex + (right ? 1 : 0);
//     events.nodeIndex[id] = nodeIndex;

    CTuint nodeIndex = src.nodeIndex[id];

    if(getAxis(bbox.m_min, splitAxis) <= split && getAxis(bbox.m_max, splitAxis) <= split)
    {
        mask[id] = 1;

        //left
        clipCopyEvent(dst, src, id, id);
        dst.nodeIndex[id] = 2 * nodeIndex;
    }
    else if(getAxis(bbox.m_min, splitAxis) >= split && getAxis(bbox.m_max, splitAxis) >= split)
    {
        mask[N + id] = 1;

        //right
        clipCopyEvent(dst, src, N + id, id);
        dst.nodeIndex[N + id] = 2 * nodeIndex + 1;
        dst.prefixSum[N + id] = src.prefixSum[id] + splits.below[isplit.index];
    }
    else
    {
        mask[id] = 1;
        mask[N + id] = 1;

        //both
        clipCopyEvent(dst, src, id, id);
        CTreal save = getAxis(bbox.m_max, splitAxis);
        setAxis(bbox.m_max, splitAxis, split);
        dst.nodeIndex[id] = 2 * nodeIndex;
        dst.ranges[id] = bbox;

        setAxis(bbox.m_max, splitAxis, save);
        clipCopyEvent(dst, src, N + id, id);
        setAxis(bbox.m_min, splitAxis, split);
        dst.nodeIndex[N + id] = 2 * nodeIndex+ 1;
        dst.ranges[N + id] = bbox;
        dst.prefixSum[N + id] = src.prefixSum[id] + splits.below[isplit.index];
    }
}

__global__ void compactEventLine(
    cuEventLine dst,
    cuEventLine src,
    const CTuint* __restrict mask,
    const CTuint* __restrict prefixSum,
    CTuint N)
{
    RETURN_IF_OOB(N);

    if(mask[id])
    {
        copyEvent(dst, src, prefixSum[id], id);
    }
}