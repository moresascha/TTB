#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"
#include <Reduce.h>

#define DYNAMIC_PARALLELISM

template <typename T>
__global__ void makeInverseScan(const T* __restrict scanVals, T* dst, CTuint N)
{
    RETURN_IF_OOB(N);

    dst[id] = id - scanVals[id];
}

__global__ void reorderEvent3(
    cuEventLineTriple dst,
    cuEventLineTriple src,
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
            events.getLine(a).type[start_index] = EVENT_START;
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

template <CTbyte LONGEST_AXIS>
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

    CTuint prefixPrims = events.lines[axis].prefixSum[id];

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
        edgeMask[id] = type == EVENT_START ? 0 : 1;
    }
    else
    {
        edgeMask[id] = type == EVENT_START ? 1 : 0;
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
    //dst.prefixSum[dstIndex] = src.prefixSum[srcIndex];
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
    const CTuint* __restrict nodeContentCount,
    Split splits,
    CTbyte myAxis,
    CTuint eventCount)
{
    RETURN_IF_OOB(eventCount);

    CTuint nodeIndex = src.nodeIndex[id];
    CTuint eventsLeftFromMe = 2 * nodeContentStart[nodeIndex];

    IndexedSAHSplit isplit = splits.indexedSplit[eventsLeftFromMe];
    
    CTbyte splitAxis = splits.axis[isplit.index];

    BBox bbox = src.ranges[id];

    CTreal v = src.indexedEvent[id].v;
    CTreal split = splits.v[isplit.index];
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
        
        //dst.prefixSum[N + id] = src.prefixSum[id] + splits.below[isplit.index];
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
        
        //dst.prefixSum[N + id] = src.prefixSum[id] + splits.below[isplit.index];

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

__global__ void initInteriorNodes(
    cuEventLineTriple events, 
    Node nodes,
    Split splits,
    const CTuint* __restrict activeNodes,
    const CTuint* __restrict activeNodesThisLevel,
    BBox* oldBBoxes,
    BBox* newBBoxes,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* newActiveNodes,
    CTbyte* activeNodesIsLeaf,
    CTuint childOffset,
    CTuint nodeOffset,
    CTuint N,
    CTuint* oldNodeContentStart,
    CTbyte makeLeaves)
{
    RETURN_IF_OOB(N);

    CTuint can = activeNodesThisLevel[id];
    CTuint an = activeNodes[id];
    CTuint edgesBeforeMe = 2 * oldNodeContentStart[can];

    IndexedSAHSplit split = splits.indexedSplit[edgesBeforeMe];

    CTbyte axis = splits.axis[split.index];
    CTreal s = splits.v[split.index];
    CTuint below = splits.below[split.index];
    CTuint above = splits.above[split.index];

    CTuint nodeId = nodeOffset + an;
    nodes.split[nodeId] = s;
    nodes.splitAxis[nodeId] = axis;

    CTuint dst = id;

    newContentCount[2 * dst + 0] = below;
    newContentCount[2 * dst + 1] = above;

//     CTuint ltmp = events.lines[axis].scannedEventTypeEndMask[edgesBeforeMe];
//     CTuint belowPrims = ltmp;
//     CTuint abovePrims = ltmp + below;
// 
//     newContentStart[2 * dst + 0] = belowPrims;
//     newContentStart[2 * dst + 1] = abovePrims;

    CTuint leftChildIndex = childOffset + 2 * can + 0;
    CTuint rightChildIndex = childOffset + 2 * can + 1;

    nodes.leftChild[nodeId] = leftChildIndex;
    nodes.rightChild[nodeId] = rightChildIndex;

    nodes.isLeaf[childOffset + 2 * can + 0] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
    nodes.isLeaf[childOffset + 2 * can + 1] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;

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
    cuEventLineTriple events,
    const CTbyte* __restrict isLeaf,
    CTuint* eventIsLeaf,
    CTuint N)
{
    RETURN_IF_OOB(N);

    eventIsLeaf[id] = isLeaf[events.lines[0].nodeIndex[id]] && (events.lines[0].type[id] == EVENT_START);
}

__global__ void compactLeafData(
    cuEventLineTriple events,
    CTuint* leafContent, 
    const CTuint* __restrict eventIsLeafMask, 
    const CTuint* __restrict prefixSumLeaf, 
    CTuint N)
{
    RETURN_IF_OOB(N);

    if(eventIsLeafMask[id])
    {
        CTuint dst = prefixSumLeaf[id];
        leafContent[dst] = events.lines[0].primId[id];
    }
}

__global__ void compactInteriorEventData(
    cuEventLineTriple dstEvent,
    cuEventLineTriple srcEvent,
    Node nodes,
    const CTbyte* __restrict activeNodeIsLeaf,
    const CTuint* __restrict interiorCountScanned,
    const CTuint* __restrict interiorContentScanned,
    const CTuint* __restrict leafContentScanned,
    CTuint N,
    CTuint leafPrimCount)
{
    RETURN_IF_OOB(N);
    CTuint oldNodeIndex = srcEvent.lines[0].nodeIndex[id];
    if(!activeNodeIsLeaf[oldNodeIndex])
    {
        //CTuint eventsBeforeMe = leafPrimCount ? 2 * nodes.contentStart[oldNodeIndex] : 0;
        CTuint eventsBeforeMe = 2 * leafContentScanned[oldNodeIndex];//leafPrimCount ? 2 * nodes.contentStart[oldNodeIndex] : 0;
        CTuint nodeIndex = interiorCountScanned[oldNodeIndex];
        for(CTbyte i = 0; i < 3; ++i)
        {
            dstEvent.lines[i].indexedEvent[id - eventsBeforeMe] = srcEvent.lines[i].indexedEvent[id];
            dstEvent.lines[i].nodeIndex[id - eventsBeforeMe] = nodeIndex;
            dstEvent.lines[i].prefixSum[id - eventsBeforeMe] = interiorContentScanned[oldNodeIndex];
            dstEvent.lines[i].primId[id - eventsBeforeMe] = srcEvent.lines[i].primId[id];
            dstEvent.lines[i].ranges[id - eventsBeforeMe] = srcEvent.lines[i].ranges[id];
            dstEvent.lines[i].type[id - eventsBeforeMe] = srcEvent.lines[i].type[id];
        }
    }
}

__global__ void setPrefixSumAndContentStart(
    cuEventLineTriple events,
    const CTuint* __restrict scannedNodesContentCount,
    CTuint* nodeContentStartAdd,
    CTuint nodesCount,
    CTuint N)
{
    RETURN_IF_OOB(N);

    if(id < nodesCount)
    {
        nodeContentStartAdd[id] = scannedNodesContentCount[id];
    }

    for(CTbyte i = 0; i < 3; ++i)
    {
        events.lines[i].prefixSum[id] = scannedNodesContentCount[events.lines[i].nodeIndex[id]];
    }
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