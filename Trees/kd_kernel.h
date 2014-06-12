#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"

#define RETURN_IF_OOB(__N) \
    CTuint id = GlobalId; \
    if(id >= __N) \
    { \
        return; \
    }

#define ONE_TO_ONE_CPY(_fieldName) dst.##_fieldName[id] = src.##_fieldName[index]

__global__ void reorderEvents(Event dst, Event src, CTuint N)
{
    RETURN_IF_OOB(N);
    CTuint index = src.indexedEdge[id].index;
    ONE_TO_ONE_CPY(nodeIndex);
    ONE_TO_ONE_CPY(prefixSum);
    ONE_TO_ONE_CPY(primId);
    ONE_TO_ONE_CPY(type);
}

__global__ void reorderSplits(Split dst, Split src, CTuint N)
{
    RETURN_IF_OOB(N);
    CTuint index = src.indexedSplit[id].index;
    ONE_TO_ONE_CPY(above);
    ONE_TO_ONE_CPY(below);
    ONE_TO_ONE_CPY(axis);
    ONE_TO_ONE_CPY(v);
}

__global__ void initNodesContent(NodeContent nodesContent, CTuint N)
{
    RETURN_IF_OOB(N);
    nodesContent.nodeIndex[id] = 0;
    nodesContent.prefixSum[id] = 0;
    nodesContent.primIndex[id] = id;
}

__global__ void createEvents(
        Event events,
        const BBox* __restrict primAxisAlignedBB,
        const BBox* __restrict nodeBBoxes,
        NodeContent nodesContent,
        CTuint N)
{
    RETURN_IF_OOB(N);
    
    CTuint node = nodesContent.nodeIndex[id];

    CTint primIndex = nodesContent.primIndex[id];

    BBox aabbs = primAxisAlignedBB[primIndex];

    BBox parentAxisAlignedBB = nodeBBoxes[node];

    CTint axis = getLongestAxis(parentAxisAlignedBB._min, parentAxisAlignedBB._max);
            
    CTuint start_index = 2 * id + 0;
    CTuint end_index = 2 * id + 1;

    events.indexedEdge[start_index].index = start_index;
    events.type[start_index] = EDGE_START;
    events.nodeIndex[start_index] = nodesContent.nodeIndex[id];
    events.primId[start_index] = nodesContent.primIndex[id];
    events.prefixSum[start_index] = 2 * nodesContent.prefixSum[id];
    events.indexedEdge[start_index].v = getAxis(aabbs._min, axis);
            
    events.indexedEdge[end_index].index = end_index;
    events.type[end_index] = EDGE_END;
    events.nodeIndex[end_index] = nodesContent.nodeIndex[id];
    events.primId[end_index] = nodesContent.primIndex[id];
    events.prefixSum[end_index] = 2 * nodesContent.prefixSum[id];
    events.indexedEdge[end_index].v = getAxis(aabbs._max, axis);
}

__global__ void computeSAHSplits(
        Event events, 
        const CTuint* __restrict nodesContentCount,
        const CTbyte* __restrict isLeaf,
        Split splits,
        const BBox* __restrict bboxes,
        NodeContent primitives,
        CTuint* scannedEdgeTypeStartMask,
        CTuint* scannedEdgeTypeEndMask,
        CTuint N) 
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = events.nodeIndex[id];

    if(isLeaf[nodeIndex])
    {
        splits.indexedSplit[id].sah = FLT_MAX;
        return;
    }

    CTbyte type = events.type[id];

    CTreal split = events.indexedEdge[id].v;

    CTuint prefixPrims = events.prefixSum[id]/2;

    CTuint primCount = nodesContentCount[nodeIndex];

    BBox bbox = bboxes[nodeIndex];

    byte axis = getLongestAxis(bbox._min, bbox._max);

    CTuint below = scannedEdgeTypeEndMask[id] - (type ^ 1) - prefixPrims;
    CTuint above = primCount - scannedEdgeTypeStartMask[id] + prefixPrims;

    splits.above[id] = above;
    splits.below[id] = below;
    splits.indexedSplit[id].index = id;
    splits.indexedSplit[id].sah = getSAH(bbox, axis, split, below, above);
    splits.axis[id] = axis;
    splits.v[id] = split;
}

__global__ void classifyEdges(
        Node nodes,
        Event edges,
        Split splits,
        CTuint* edgeMask,
        CTuint nodeOffset,
        CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = edges.nodeIndex[id];

    CTbyte type = edges.type[id];

    if(*(nodes.isLeaf + nodeOffset + nodeIndex))
    {
        edgeMask[id] = type == EDGE_START;
        return;
    }

    CTuint edgesBeforeMe = 2 * nodes.contentStart[nodeIndex];
    IndexedSAHSplit is = splits.indexedSplit[edgesBeforeMe];

    CTreal split = splits.v[is.index];

    CTreal v = edges.indexedEdge[id].v;

    CTuint right = !(split > v || (v == split && type == EDGE_END));
            
    CTuint nextPos = 2 * nodeIndex;
            
    if(right)
    {
        edgeMask[id] = type == EDGE_START ? 0 : 1;
    }
    else
    {
        edgeMask[id] = type == EDGE_START ? 1 : 0;
    }
            
    nodeIndex = nextPos + (right ? 1 : 0);
    edges.nodeIndex[id] = nodeIndex;
}

__global__ void compactLeafNInteriorContent(CTuint* leafContent, 
                                            CTuint* interiorContent, 
                                            const CTuint* __restrict leafCountScan,
                                            const CTbyte* __restrict isLeafMask, 
                                            const CTuint* __restrict prefixSumLeaf, 
                                            const CTuint* __restrict prefixSumInterior,
                                            const CTuint* __restrict noLeafPrefixSum,

                                            const CTuint* __restrict primIds,
                                            const CTuint* __restrict oldPrimNodeIndex,
                                            const CTuint* __restrict oldPrimPrefixSum,

                                            CTuint* newPrimNodeIndex,
                                            CTuint* newPrimPrefixSum,
                                            CTuint N)
{
    RETURN_IF_OOB(N);

    if(isLeafMask[id])
    {
        CTuint dst = prefixSumLeaf[id];
        leafContent[dst] = primIds[id];
    }
    else
    {
        CTuint nodeIndex = oldPrimNodeIndex[id];
        //nodeIndex -= leafCountScan[nodeIndex/2 + nodeIndex%2];
        nodeIndex = noLeafPrefixSum[nodeIndex/2 + nodeIndex%2];
        CTuint dst = prefixSumInterior[id];
        interiorContent[dst] = primIds[id];
        newPrimNodeIndex[dst] = nodeIndex;
        newPrimPrefixSum[dst] = oldPrimPrefixSum[id];
    }
}

__global__ void setPrimBelongsToLeaf(
    Event events,
    NodeContent nodesContent,
    const CTbyte* __restrict isLeaf,
    const CTuint* __restrict mask,
    const CTuint* __restrict scanned,
    CTuint depth,
    CTuint N)
{
    RETURN_IF_OOB(N);

    if(mask[id])
    {
        CTuint nodeIndex = events.nodeIndex[id];
        nodesContent.primIsLeaf[scanned[id]] = isLeaf[nodeIndex];
    }
}

__global__ void fixContentStartAddr(
    Node nodes,
    const CTuint* __restrict isLeafScanned,
    const CTbyte* __restrict isLeaf,
    CTuint depth,
    CTuint N)
{
    RETURN_IF_OOB(N);

    if(isLeaf[id])
    {
        return;
    }

    CTuint startAdd = nodes.contentStart[2 * id + 0];
    nodes.contentStart[2 * id + 0] = startAdd - isLeafScanned[startAdd];

    startAdd = nodes.contentStart[2 * id + 1];
    nodes.contentStart[2 * id + 1] = startAdd - isLeafScanned[startAdd];
}

__global__ void createContentMasks(
    const CTbyte* __restrict isLeaf,
    const CTuint* __restrict content,
    CTuint* leafMask, 
    CTuint* interiorMask,
    CTuint N
    )
{
    RETURN_IF_OOB(N);
    leafMask[id] = isLeaf[id] * content[id];
    interiorMask[id] = (1 ^ isLeaf[id]) * content[id];
}

template < 
    CTbyte LAST_LEVEL
>
__global__ void compactLeafNInteriorData(
    const CTuint* __restrict scannedInteriorContentstart,
    const CTuint* __restrict scannedLeafContentstart,
    const CTuint* __restrict nodesContentCount,
    const BBox* __restrict nodeBBoxes,
    const CTbyte* __restrict isLeaf, 
    const CTuint* __restrict leafCountScan, 
    const CTuint* __restrict interiorCountScan,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* leafContentStart, 
    CTuint* leafContentCount,
    BBox* newBBxes,
    CTuint depth, 
    CTuint lastLeafCount,
    CTuint contentStartOffset,
    CTuint N)
{
    RETURN_IF_OOB(N);

    if(LAST_LEVEL && id < N/2)
    {
        leafContentStart[lastLeafCount + id] = contentStartOffset + scannedLeafContentstart[id];
        leafContentCount[lastLeafCount + id] = nodesContentCount[id];
        return;
    }

    if(id < N/2)
    {
        if(isLeaf[id])
        {
            CTuint dst = leafCountScan[id];
            leafContentStart[lastLeafCount + dst] = contentStartOffset + scannedLeafContentstart[id];
            leafContentCount[lastLeafCount + dst] = nodesContentCount[id];
        }
    }
    else
    {
        if(isLeaf[id/2])
        {
            CTuint dst = interiorCountScan[id/2];
            dst += id%2;
            newContentCount[dst] = nodesContentCount[id];
            newContentStart[dst] = scannedInteriorContentstart[id];
            newBBxes[dst] = nodeBBoxes[id];
        }
    }
}

__global__ void makeLeafFromIfMaxPrims(
    CTuint* nodesContentCount,
    CTbyte* isLeaf, 
    CTuint maxPrims, 
    CTuint N)
{
    RETURN_IF_OOB(N);
    isLeaf[id] = nodesContentCount[id] <= maxPrims ? 1 : 0;
}

__global__ void makeLealIfBadSplit(
    Node nodes,
    CTbyte* isLeaf,
    Split splits,
    CTuint N
    )
{
    RETURN_IF_OOB(N);
    isLeaf[id] = splits.indexedSplit[2 * nodes.contentStart[id]].sah == FLT_MAX;
}

__global__ void compactPrimitivesFromEvents(
        Event events,
        Node nodes,
        NodeContent nodesContent,
        const CTuint* __restrict mask,
        const CTuint* __restrict scanned,
        CTuint depth,
        CTuint N)
{
    RETURN_IF_OOB(N);

    if(mask[id])
    {
        CTuint dst = scanned[id];
        CTuint nodeIndex = events.nodeIndex[id];
        nodesContent.nodeIndex[dst] = nodeIndex;
        nodesContent.primIndex[dst] = events.primId[id];
        nodesContent.prefixSum[dst] = nodes.contentStart[nodeIndex];
    }
}

void __device__ splitAABB(BBox* aabb, CTreal split, CTbyte axis, BBox* l, BBox* r)
{
    l->_max = aabb->_max;
    l->_min = aabb->_min;
    r->_max = aabb->_max;
    r->_min = aabb->_min;
    switch(axis)
    {
    case 0:
        {
            l->_max.x = split; r->_min.x = split; 
        } break;
    case 1:
        {
            l->_max.y = split; r->_min.y = split; 
        } break;
    case 2:
        {
            l->_max.z = split; r->_min.z = split; 
        } break;
    }
}

__global__ void initNodes(
    Node nodes,
    Split splits,
    CTuint* leafNodesContentCount,
    CTuint* leafNodesContentStart,
    const CTuint* __restrict scannedEdges,
    const CTuint* __restrict leafScanned,
    const CTuint* __restrict interiorScanned,
    const CTuint* __restrict isPrimLeafScanned,
    BBox* oldBBoxes,
    BBox* newBBoxes,
    CTuint interiorNodeOffset,
    CTuint currentLeafNodes,
    CTuint nodeOffset,
    CTuint gotLeaves,
    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeId = id + nodeOffset;
    CTbyte isMeLeaf = nodes.isLeaf[nodeId];

    if(isMeLeaf)
    {
        CTuint dst = leafScanned[id];
        leafNodesContentCount[currentLeafNodes + dst] = nodes.contentCount[id];
        leafNodesContentStart[currentLeafNodes + dst] = nodes.contentStart[id];
        return;
    }

    CTuint edgesBeforeMe = 2 * nodes.contentStart[id];
    IndexedSAHSplit split = splits.indexedSplit[edgesBeforeMe];

    CTreal s = splits.v[split.index];
    CTbyte axis = splits.axis[split.index];
    nodes.split[nodeId] = s;
    nodes.splitAxis[nodeId] = axis;

    CTuint above = splits.above[split.index];
    CTuint below = splits.below[split.index];

    CTuint dst = gotLeaves ? interiorScanned[id] : 0;

    nodes.contentCount[2 * dst + 0] = below;
    nodes.contentCount[2 * dst + 1] = above;

    CTuint ltmp = scannedEdges[edgesBeforeMe];
    CTuint belowPrims = ltmp;
    CTuint abovePrims = ltmp + below;

    if(gotLeaves)
    {
        belowPrims = belowPrims - isPrimLeafScanned[belowPrims];
        abovePrims = abovePrims - isPrimLeafScanned[abovePrims];
    }

    nodes.contentStart[2 * dst + 0] = belowPrims;
    nodes.contentStart[2 * dst + 1] = abovePrims;

    CTuint childOffset = interiorNodeOffset + currentLeafNodes;
    CTuint leftChildIndex = childOffset + 2 * id + 0;
    CTuint rightChildIndex = childOffset + 2 * id + 1;

    nodes.leftChild[nodeId] = leftChildIndex;
    nodes.rightChild[nodeId] = rightChildIndex;

    BBox l;
    BBox r;

    splitAABB(&oldBBoxes[id], s, axis, &l, &r);
    newBBoxes[2 * dst + 0] = l;
    newBBoxes[2 * dst + 1] = r;
}

#if defined DYNAMIC_PARALLELISM

__global__ void startSort(Node nodes, IndexedEdge* edges, uint offset);

__global__ void startGetMinSplits(Node nodes, IndexedSplit* splits, uint offset);

#endif