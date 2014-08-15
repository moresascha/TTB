#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"


#if 0

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

    events.type[start_index] = EVENT_START;
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

    CTint axis = getLongestAxis(parentAxisAlignedBB.m_min, parentAxisAlignedBB.m_max);

    events.indexedEdge[start_index].v = getAxis(aabbs.m_min, axis);

    events.indexedEdge[end_index].v = getAxis(aabbs.m_max, axis);
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

    byte axis = getLongestAxis(bbox.m_min, bbox.m_max);

    CTuint above = primCount - scannedEdgeTypeEndMask[id] + prefixPrims - type;
    CTuint below = scannedEdgeTypeStartMask[id] - prefixPrims;

    splits.above[id] = above;
    splits.below[id] = below;
    splits.indexedSplit[id].index = id;
    //splits.indexedSplit[id].sah = (bla>=2 && nodeIndex==2)? INVALID_SAH : getSAH(bbox, axis, split, below, above);
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
        const CTuint* contentStartAdd,
        CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = events.nodeIndex[id];

    if(isLeaf[nodeIndex])
    {
        edgeMask[id] = 0;
        events.nodeIndex[id] = (CTuint)-1;
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

__global__ void setPrimBelongsToLeaf(
    NodeContent nodesContent,
    const CTbyte* __restrict isLeaf,
    CTuint N)
{
    RETURN_IF_OOB(N);

    CTuint nodeIndex = nodesContent.nodeIndex[id];
    nodesContent.primIsLeaf[id] = isLeaf[nodeIndex] > 0;
}

__global__ void setPrimBelongsToLeafFromEvents(
    Event events,
    NodeContent nodesContent,
    const CTbyte* __restrict isLeaf,
    const CTuint* __restrict mask,
    const CTuint* __restrict scanned,
    CTuint N, CTuint* blbla)
{
    RETURN_IF_OOB(N);

    if(mask[id])
    {
        CTuint nodeIndex = events.nodeIndex[id];
        nodesContent.primIsLeaf[scanned[id]] = isLeaf[nodeIndex] > 0;
        blbla[scanned[id]] = nodeIndex;
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
    CTbyte mask = isLeaf[id];
    leafMask[id] = (mask == 1) * contentCount[id];
    interiorMask[id] = (mask < 2) * (1 ^ mask) * contentCount[id];
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
    const CTuint* __restrict activeNodes,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* leafContentStart, 
    CTuint* leafContentCount,
    CTuint* newActiveNodes,
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
    CTbyte leafMask = isLeaf[id];
    CTuint dst = leafCountScan[id];
    if(leafMask && leafMask < 2)
    {
        leafContentStart[leafCount + dst] = contentStartOffset + scannedLeafContent[id];
        leafContentCount[leafCount + dst] = nodesContentCount[id];
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

template <CTbyte init>
__global__ void setNodeToLeafIndex(
    CTuint* nodeToLeafIndex,
    const CTuint* __restrict activeNodes,
    const CTuint* __restrict isLeafScan,
    const CTbyte* __restrict isLeafMask, 
    CTuint offset,
    CTuint leafCount,
    CTuint N
    )
{
    RETURN_IF_OOB(N);
    if(init == 2)
    {
        nodeToLeafIndex[offset + id] = 0;
        return;
    }
    CTbyte isLeaf = isLeafMask[id];
    if(isLeaf && isLeaf < 2)
    {
        if(init)
        {
            nodeToLeafIndex[activeNodes[id] + offset] = leafCount + isLeafScan[id];
        }
        else
        {
            nodeToLeafIndex[activeNodes[id] + offset] += leafCount + isLeafScan[id];
        }
    }
}

__global__ void makeLeafIfBadSplitOrLessThanMaxElements(
    Node nodes,
    CTbyte* nodeIsLeaf,
    CTuint* ativeNodes,
    CTbyte* isLeaf,
    Split splits,
    CTbyte makeChildLeaves,
    CTuint N
    )
{
    RETURN_IF_OOB(N);
    CTuint splitAdd = 2 * nodes.contentStart[id];
    IndexedSAHSplit split = splits.indexedSplit[splitAdd];
    CTbyte isBad = IS_INVALD_SAH(split.sah);
    isLeaf[id] = isBad;
    if(isBad)
    {
        nodeIsLeaf[ativeNodes[id]] = 1;
    }
    CTuint below = splits.below[split.index];
    CTuint above = splits.above[split.index];
    isLeaf[N + 2 * id + 0] = isBad ? 2 : ((below <= MAX_ELEMENTS_PER_LEAF) || makeChildLeaves);
    isLeaf[N + 2 * id + 1] = isBad ? 2 : ((above <= MAX_ELEMENTS_PER_LEAF) || makeChildLeaves);
}

__global__ void compactPrimitivesFromEvents(
        Event events,
        Node nodes,
        NodeContent nodesContent,
        const CTuint* __restrict leafScan,
        const CTuint* __restrict mask,
        const CTuint* __restrict scanned,
        CTuint depth,
        CTuint N,
        CTuint blabla)
{
    RETURN_IF_OOB(N);

    if(mask[id])
    {
        CTuint dst = scanned[id];
        CTuint nodeIndex = events.nodeIndex[id];
        CTuint parent = nodeIndex/2;
        CTuint prefixLeaves = leafScan[parent];
        CTuint newNodeIndex = nodeIndex - 2 * prefixLeaves;
        nodesContent.nodeIndex[dst] = newNodeIndex;
        nodesContent.primIndex[dst] = events.primId[id];
        nodesContent.prefixSum[dst] = nodes.contentStart[newNodeIndex];
        //blabla[dst] = id;
    }
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
    const CTuint* __restrict activeNodesThisLevel,
    BBox* oldBBoxes,
    BBox* newBBoxes,
    CTuint* newContentCount,
    CTuint* newContentStart,
    CTuint* newActiveNodes,
    CTbyte* activeNodesIsLeaf,
    CTuint childOffset,
    CTuint childOffset2,
    CTuint nodeOffset,
    CTuint gotLeaves,
    CTuint N,
    CTuint* oldNodeContentStart,
    CTbyte makeLeaves)
{
    RETURN_IF_OOB(N);

    CTuint an = activeNodes[id];
    CTuint can = activeNodesThisLevel[id];
    CTuint nodeId = nodeOffset + an;

//     if(nodes.isLeaf[nodeOffset + activeNodes[id]])
//     {
//         activeNodesIsLeaf[2 * id + 0] = 2;
//         activeNodesIsLeaf[2 * id + 1] = 2;
//         return;
//     }

    CTuint edgesBeforeMe = 2 * oldNodeContentStart[can];
    IndexedSAHSplit split = splits.indexedSplit[edgesBeforeMe];

    CTreal s = splits.v[split.index];
    CTbyte axis = splits.axis[split.index];
    nodes.split[nodeId] = s;
    nodes.splitAxis[nodeId] = axis;

    CTuint below = splits.below[split.index];
    CTuint above = splits.above[split.index];

    CTuint dst = id; //gotLeaves ? interiorNodesScan[id] : id;

    newContentCount[2 * dst + 0] = below;
    newContentCount[2 * dst + 1] = above;

    CTuint ltmp = scannedEdges[edgesBeforeMe];
    CTuint belowPrims = ltmp;
    CTuint abovePrims = ltmp + below;

    newContentStart[2 * dst + 0] = belowPrims;
    newContentStart[2 * dst + 1] = abovePrims;

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

#if defined DYNAMIC_PARALLELISM

__global__ void startSort(Node nodes, IndexedEdge* edges, uint offset);

__global__ void startGetMinSplits(Node nodes, IndexedSplit* splits, uint offset);

#endif

#endif