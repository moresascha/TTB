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
    CTuint start_index = 2 * id + 0;
    CTuint end_index = 2 * id + 1;

    events.type[start_index] = EDGE_START;
    events.type[end_index] = EDGE_END;

    events.nodeIndex[start_index] = nodesContent.nodeIndex[id];
    events.nodeIndex[end_index] = nodesContent.nodeIndex[id];

    events.indexedEdge[start_index].index = start_index;
    events.indexedEdge[end_index].index = end_index;

    events.primId[start_index] = nodesContent.primIndex[id];
    events.prefixSum[start_index] = 2 * nodesContent.prefixSum[id];

    events.primId[end_index] = nodesContent.primIndex[id];
    events.prefixSum[end_index] = 2 * nodesContent.prefixSum[id];

    CTint primIndex = nodesContent.primIndex[id];

    BBox aabbs = primAxisAlignedBB[primIndex];

    BBox parentAxisAlignedBB = nodeBBoxes[node];

    CTint axis = getLongestAxis(parentAxisAlignedBB._min, parentAxisAlignedBB._max);

    events.indexedEdge[start_index].v = getAxis(aabbs._min, axis);

    events.indexedEdge[end_index].v = getAxis(aabbs._max, axis);
}

__global__ void computeSAHSplits(
        Event events, 
        const CTuint* __restrict nodesContentCount,
        Split splits,
        const BBox* __restrict bboxes,
        NodeContent primitives,
        const CTuint* __restrict scannedEdgeTypeStartMask,
        const CTuint* __restrict scannedEdgeTypeEndMask,
        CTuint N,
        CTuint bla) 
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = events.nodeIndex[id];

    CTbyte type = events.type[id];

    CTreal split = events.indexedEdge[id].v;

    CTuint prefixPrims = events.prefixSum[id]/2;

    CTuint primCount = nodesContentCount[nodeIndex];

    BBox bbox = bboxes[nodeIndex];

    byte axis = getLongestAxis(bbox._min, bbox._max);

    CTuint below = scannedEdgeTypeStartMask[id]  - prefixPrims;
    CTuint above = primCount - scannedEdgeTypeEndMask[id] + prefixPrims - type;

    splits.above[id] = above;
    splits.below[id] = below;
    splits.indexedSplit[id].index = id;
    splits.indexedSplit[id].sah = getSAH(bbox, axis, split, below, above);
    splits.axis[id] = axis;
    splits.v[id] = split;
}

__global__ void classifyEdges(
        Node nodes,
        Event events,
        Split splits,
        const CTbyte* isLeaf,
        CTuint* edgeMask,
        CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = events.nodeIndex[id];

    CTbyte type = events.type[id];

    CTuint edgesBeforeMe = 2 * nodes.contentStart[nodeIndex];
    IndexedSAHSplit is = splits.indexedSplit[edgesBeforeMe];

    CTreal split = splits.v[is.index];

    CTreal v = events.indexedEdge[id].v;

    CTuint right = !(split > v || (v == split && type == EDGE_END));

    if(isLeaf[nodeIndex])
    {
        edgeMask[id] = 0;
        return;
    }
            
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

__global__ void setPrimBelongsToLeaf(
    NodeContent nodesContent,
    const CTuint* __restrict activeNodes,
    const CTbyte* __restrict isLeaf,
    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = nodesContent.nodeIndex[id];
    nodesContent.primIsLeaf[id] = isLeaf[nodeIndex];
}

__global__ void setPrimBelongsToLeafFromEvents(
    Event events,
    NodeContent nodesContent,
    const CTbyte* __restrict isLeaf,
    const CTuint* __restrict mask,
    const CTuint* __restrict scanned,
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
    CTuint N)
{
    RETURN_IF_OOB(N);

    if(isLeaf[id])
    {
        return;
    }

    CTuint startAdd = nodes.contentStart[id];
    nodes.contentStart[id] = startAdd - isLeafScanned[startAdd];
}

__global__ void createInteriorAndLeafContentCountMasks(
    const CTbyte* __restrict isLeaf,
    const CTuint* __restrict contentCount,
    CTuint* leafMask, 
    CTuint* interiorMask,
    CTuint N)
{
    RETURN_IF_OOB(N);
    leafMask[id] = isLeaf[id] * contentCount[id];
    interiorMask[id] = (1 ^ isLeaf[id]) * contentCount[id];
}

__global__ void compactLeafNInteriorContent(
    CTuint* leafContent, 
    CTuint* interiorContent,
    const CTbyte* __restrict isLeafMask, 
    const CTuint* __restrict prefixSumLeaf, 
    const CTuint* __restrict prefixSumInterior,
    const CTuint* __restrict noLeafPrefixSum,

    const CTuint* __restrict primIds,
    const CTuint* __restrict oldPrimNodeIndex,
    const CTuint* __restrict scannedContentStart,

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
        CTuint oldNodeIndex = oldPrimNodeIndex[id];
        CTuint nodeIndex = noLeafPrefixSum[oldNodeIndex];
        CTuint dst = prefixSumInterior[id];
        interiorContent[dst] = primIds[id];
        newPrimNodeIndex[dst] = nodeIndex;
        newPrimPrefixSum[dst] = scannedContentStart[oldNodeIndex];
    }
}

__global__ void compactLeafNInteriorData(
    const CTuint* __restrict scannedInteriorContent,
    const CTuint* __restrict scannedLeafContent,
    const CTuint* __restrict nodesContentCount,
    const CTuint* __restrict nodesContentStart,
    const BBox* __restrict nodeBBoxes,
    const CTbyte* __restrict isLeaf, 
    const CTuint* __restrict primIsLeafScanned,
    const CTuint* __restrict leafCountScan, 
    const CTuint* __restrict interiorCountScan,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* leafContentStart, 
    CTuint* leafContentCount,
    CTuint* activeNodes,
    BBox* newBBxes,
    CTuint leafCount,
    CTuint contentStartOffset,
    CTuint N)
{
    RETURN_IF_OOB(N);

//     if(LAST_LEVEL)
//     {
//         leafContentStart[lastLeafCount + id] = contentStartOffset + scannedLeafContentstart[id];
//         leafContentCount[lastLeafCount + id] = nodesContentCount[id];
//         return;
//     }

    if(isLeaf[id])
    {
        CTuint dst = leafCountScan[id];
        leafContentStart[leafCount + dst] = contentStartOffset + scannedLeafContent[id];
        leafContentCount[leafCount + dst] = nodesContentCount[id];
    }
    else
    {
        CTuint dst = interiorCountScan[id];
        activeNodes[dst] = id;
        newContentCount[dst] = nodesContentCount[id];
        newContentStart[dst] = scannedInteriorContent[id]; //scannedInteriorContentstart[id];
        newBBxes[dst] = nodeBBoxes[id];
    }
}

__global__ void makeLealIfBadSplitOrLessThanMaxElements(
    Node nodes,
    CTbyte* nodeIsLeaf,
    CTbyte* isLeaf,
    Split splits,
    CTbyte makeChildLeaves,
    CTuint N
    )
{
    RETURN_IF_OOB(N);
    CTuint splitAdd = 2 * nodes.contentStart[id];
    CTbyte isBad = IS_INVALD_SAH(splits.indexedSplit[splitAdd].sah);
    isLeaf[id] = isBad;
    if(isBad)
    {
        nodeIsLeaf[id] = 1;
    }
    isLeaf[N + 2 * id + 0] = splits.below[splits.indexedSplit[splitAdd].index] <= MAX_ELEMENTS_PER_LEAF || makeChildLeaves;
    isLeaf[N + 2 * id + 1] = splits.above[splits.indexedSplit[splitAdd].index] <= MAX_ELEMENTS_PER_LEAF || makeChildLeaves;
}

__global__ void compactPrimitivesFromEvents(
        Event events,
        Node nodes,
        NodeContent nodesContent,
        const CTuint* __restrict leafScan,
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
        CTuint prefixLeaves = leafScan[nodeIndex/2];
        nodesContent.nodeIndex[dst] = nodeIndex - 2 * prefixLeaves;
        nodesContent.primIndex[dst] = events.primId[id];
        nodesContent.prefixSum[dst] = nodes.contentStart[nodeIndex - 2 * prefixLeaves];
    }
}

template <CTbyte hasLeaves>
__global__ void setActiveNodesMask(
    CTuint* ids,
    const CTbyte* __restrict isLeaf,
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

void __device__ splitAABB(const BBox* aabb, CTreal split, CTbyte axis, BBox* l, BBox* r)
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

__global__ void setNewContentCountAndContentStartAdd(
    Node nodes,
    Split splits,
    CTuint* leafNodesContentCount,
    CTuint* leafNodesContentStart,
    const CTuint* __restrict scannedEdges,
    const CTuint* __restrict interiorNodesScan,
    const CTuint* __restrict isPrimLeafScanned,
    const CTuint* __restrict activeNodes,
    const CTbyte* __restrict activeNodesLeaf,
    BBox* oldBBoxes,
    BBox* newBBoxes,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint childOffset,
    CTuint nodeOffset,
    CTuint gotLeaves,
    CTuint N)
{

}

__global__ void initThisLevelInteriorNodes(
    Node nodes,
    Split splits,
    CTuint* leafNodesContentCount,
    CTuint* leafNodesContentStart,
    const CTuint* __restrict scannedEdges,
    const CTuint* __restrict interiorNodesScan,
    const CTuint* __restrict isPrimLeafScanned,
    const CTuint* __restrict activeNodes,
    const CTbyte* __restrict activeNodesLeaf,
    BBox* oldBBoxes,
    BBox* newBBoxes,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint childOffset,
    CTuint nodeOffset,
    CTuint gotLeaves,
    CTuint N,
    CTuint* oldNodeContentStart,
    CTbyte makeLeaves)
{
    RETURN_IF_OOB(N);

    CTuint nodeId = nodeOffset + activeNodes[id];

    CTuint edgesBeforeMe = 2 * oldNodeContentStart[gotLeaves ? activeNodes[id] : id]; //nodes.contentStart[id];
    IndexedSAHSplit split = splits.indexedSplit[edgesBeforeMe];

    CTreal s = splits.v[split.index];
    CTbyte axis = splits.axis[split.index];
    //nodes.isLeaf[nodeId] = activeNodesLeaf[activeNodes[id]];
    nodes.split[nodeId] = s;
    nodes.splitAxis[nodeId] = axis;

    CTuint above = splits.above[split.index];
    CTuint below = splits.below[split.index];

    CTuint dst = gotLeaves ? interiorNodesScan[id] : id;

    newContentCount[2 * dst + 0] = below;
    newContentCount[2 * dst + 1] = above;

    CTuint ltmp = scannedEdges[edgesBeforeMe];
    CTuint belowPrims = ltmp;
    CTuint abovePrims = ltmp + below;

    if(gotLeaves)
    {
       // belowPrims = belowPrims - isPrimLeafScanned[belowPrims];
       // abovePrims = abovePrims - isPrimLeafScanned[abovePrims];
    }

    newContentStart[2 * dst + 0] = belowPrims;
    newContentStart[2 * dst + 1] = abovePrims;

    CTuint leftChildIndex = childOffset + 2 * id + 0;
    CTuint rightChildIndex = childOffset + 2 * id + 1;

    nodes.leftChild[nodeId] = leftChildIndex;
    nodes.rightChild[nodeId] = rightChildIndex;

    nodes.isLeaf[leftChildIndex] = below <= MAX_ELEMENTS_PER_LEAF || makeLeaves;
    nodes.isLeaf[rightChildIndex] = above <= MAX_ELEMENTS_PER_LEAF || makeLeaves;

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

#if defined DYNAMIC_PARALLELISM

__global__ void startSort(Node nodes, IndexedEdge* edges, uint offset);

__global__ void startGetMinSplits(Node nodes, IndexedSplit* splits, uint offset);

#endif