#pragma once
#include "vec_functions.h"
#include "cuKDTree.h"

#define RETURN_IF_OOB(__N) \
    uint id = GlobalId; \
    if(id >= __N) \
    { \
        return; \
    }

#define ONE_TO_ONE_CPY(_fieldName) dst.##_fieldName[id] = src.##_fieldName[index]

__global__ void reorderEdges(Edge dst, Edge src, uint N)
{
    RETURN_IF_OOB(N);
    uint index = src.indexedEdge[id].index;
    ONE_TO_ONE_CPY(nodeIndex);
    ONE_TO_ONE_CPY(prefixSum);
    ONE_TO_ONE_CPY(primId);
    ONE_TO_ONE_CPY(type);
}

__global__ void reorderSplits(Split dst, Split src, uint N)
{
    RETURN_IF_OOB(N);
    uint index = src.indexedSplit[id].index;
    ONE_TO_ONE_CPY(above);
    ONE_TO_ONE_CPY(below);
    ONE_TO_ONE_CPY(axis);
    ONE_TO_ONE_CPY(v);
}

__global__ void initNodesContent(Primitive nodesContent, uint N)
{
    RETURN_IF_OOB(N);
    nodesContent.nodeIndex[id] = 0;
    nodesContent.prefixSum[id] = 0;
    nodesContent.primIndex[id] = id;
}

__global__ void createEdges(
        Edge edges, 
        Node nodes, 
        BBox* primAxisAlignedBB,
        Primitive nodesContent,
        byte depth, 
        uint N)
{
    RETURN_IF_OOB(N);
    
    uint offset = (1 << depth) - 1;

    int primIndex = nodesContent.primIndex[id];

    BBox aabbs = primAxisAlignedBB[primIndex];

    BBox parentAxisAlignedBB = nodes.aabb[offset + nodesContent.nodeIndex[id]];

    int axis = getLongestAxis(parentAxisAlignedBB._min, parentAxisAlignedBB._max);
            
    uint start_index = 2 * id + 0;
    uint end_index = 2 * id + 1;

    edges.indexedEdge[start_index].index = start_index;
    edges.type[start_index] = EDGE_START;
    edges.nodeIndex[start_index] = nodesContent.nodeIndex[id];
    edges.primId[start_index] = nodesContent.primIndex[id];
    edges.prefixSum[start_index] = 2 * nodesContent.prefixSum[id];
    edges.indexedEdge[start_index].v = getAxis(aabbs._min, axis);
            
    edges.indexedEdge[end_index].index = end_index;
    edges.type[end_index] = EDGE_END;
    edges.nodeIndex[end_index] = nodesContent.nodeIndex[id];
    edges.primId[end_index] = nodesContent.primIndex[id];
    edges.prefixSum[end_index] = 2 * nodesContent.prefixSum[id];
    edges.indexedEdge[end_index].v = getAxis(aabbs._max, axis);
}

__global__ void computeSAHSplits(
        Edge edges, 
        Node nodes,
        Split splits,
        Primitive primitives,
        uint* scannedEdgeTypeStartMask,
        uint* scannedEdgeTypeEndMask,
        byte depth, 
        uint N) 
{
    RETURN_IF_OOB(N);
    uint lelvelOffset = (1 << depth) - 1;

    byte type = edges.type[id];

    float split = edges.indexedEdge[id].v;

    uint nodeIndex = lelvelOffset + edges.nodeIndex[id];

    int prefixPrims = edges.prefixSum[id]/2;

    int primCount = nodes.contentCount[nodeIndex];

    BBox bbox = nodes.aabb[nodeIndex];

    byte axis = getLongestAxis(bbox._min, bbox._max);

    uint above = primCount - scannedEdgeTypeStartMask[id] - prefixPrims;
    uint below = scannedEdgeTypeEndMask[id] - (type ^ 1) - prefixPrims;

    splits.above[id] = above;
    splits.below[id] = below;
    splits.indexedSplit[id].index = id;
    splits.indexedSplit[id].sah = getSAH(nodes.aabb[nodeIndex], axis, split, below, above);
    splits.axis[id] = axis;
    splits.v[id] = split;
}

__global__ void classifyEdges(
        Node nodes,
        Edge edges,
        uint* edgeMask,
        byte depth,
        uint N)
{
    RETURN_IF_OOB(N);

    uint offset = (1 << depth) - 1;

    uint nodeIndex = edges.nodeIndex[id];

    float split = nodes.split[offset + nodeIndex];

    float v = edges.indexedEdge[id].v;

    int right = !(split > v || (v == split && edges.type[id] == EDGE_END));
            
    uint nextPos = 2 * nodeIndex;
            
    if(right)
    {
        edgeMask[id] = edges.type[id] == EDGE_START ? 0 : 1;
    }
    else
    {
        edgeMask[id] = edges.type[id] == EDGE_START ? 1 : 0;
    }
            
    nodeIndex = nextPos + (right ? 1 : 0);
    edges.nodeIndex[id] = nodeIndex;
}

__global__ void compactContentFromEdges(
        Edge edges,
        Node nodes,
        Primitive nodesContent,
        const uint* __restrict mask,
        const uint* __restrict scan,
        uint depth,
        uint N)
{
    RETURN_IF_OOB(N);
    if(mask[id])
    {
        uint offset = (1 << (depth+1)) - 1;
        uint meStartCount = nodes.contentStart[offset + edges.nodeIndex[id]];
        uint dst = scan[id];
        nodesContent.nodeIndex[dst] = edges.nodeIndex[id];
        nodesContent.primIndex[dst] = edges.primId[id];
        nodesContent.prefixSum[dst] = meStartCount;
    }
}

void __device__ splitAABB(BBox* aabb, float split, byte axis, BBox* l, BBox* r)
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

template <byte FORCE_LEAF>
__global__ void initNodes(
    Node nodes,
    Split splits,
    uint* scannedEdges,
    uint depth)
{
    uint id = GlobalId;
    uint offset = (1 << depth) - 1;
    uint index = id + offset;
    
    byte leaf = nodes.isLeaf[index];

    uint leftChildIndex = (1 << (depth+1)) - 1 + 2 * id + 0;
    uint rightChildIndex = (1 << (depth+1)) - 1 + 2 * id + 1;

    if(leaf)
    {
        //nodes.isLeaf[leftChildIndex] = 1;
        //nodes.isLeaf[rightChildIndex] = 1;
        return;
    }

    uint edgesBeforeMe = 2 * nodes.contentStart[index];
    IndexedSAHSplit split = splits.indexedSplit[edgesBeforeMe];

    byte axis = splits.axis[split.index];
    byte isParentLeaf = nodes.isLeaf[index];

    BBox l;
    BBox r;

    float s = splits.v[split.index];

    splitAABB(&nodes.aabb[index], s, axis, &l, &r);

    if(!FORCE_LEAF)
    {
        nodes.aabb[leftChildIndex] = l;
        nodes.aabb[rightChildIndex] = r;
    }

    nodes.split[index] = s;
    nodes.splitAxis[index] = axis;

    uint above = splits.above[split.index];
    uint below = splits.below[split.index];

    nodes.contentCount[leftChildIndex] = below;
    nodes.contentCount[rightChildIndex] = above;

    nodes.isLeaf[leftChildIndex] = FORCE_LEAF || below < MAX_ELEMENTS_PER_LEAF;
    nodes.isLeaf[rightChildIndex] = FORCE_LEAF || above < MAX_ELEMENTS_PER_LEAF;
    
    uint ltmp = scannedEdges[edgesBeforeMe];
    nodes.contentStart[leftChildIndex] = ltmp;
    nodes.contentStart[rightChildIndex] = ltmp + below;
}

template <byte FORCE_LEAF>
__global__ void initCurrentNodesAndCreateChilds(
        const uint* __restrict scannedEdgeMask,
        const uint* __restrict edgeMask,
        Node nodes, 
        uint depth)
{
    uint id = GlobalId;
    uint offset = (1 << depth) - 1;
    uint parentIndex = id + offset;

    byte axis = nodes.splitAxis[parentIndex];
    byte isParentLeaf = nodes.isLeaf[parentIndex];

    uint leftChildIndex = (1 << (depth+1)) - 1 + 2 * id + 0;
    uint rightChildIndex = (1 << (depth+1)) - 1 + 2 * id + 1;

    if(isParentLeaf)
    {
        nodes.isLeaf[leftChildIndex] = 1;
        nodes.isLeaf[rightChildIndex] = 1;
        return;
    }

    uint edgesBeforeMe = 2 * nodes.contentStart[parentIndex];

    int below = (int)nodes.below[parentIndex];
    int above = (int)nodes.above[parentIndex];

    BBox l;
    BBox r;

    float s = nodes.split[parentIndex];

    splitAABB(&nodes.aabb[parentIndex], s, axis, &l, &r);

    if(!FORCE_LEAF)
    {
        nodes.aabb[leftChildIndex] = l;
        nodes.aabb[rightChildIndex] = r;
    }

    uint os = scannedEdgeMask[edgesBeforeMe + max(0, below)];
    uint elemsleft = scannedEdgeMask[edgesBeforeMe + max(0, below)] - scannedEdgeMask[edgesBeforeMe];
    uint elemsright = scannedEdgeMask[max(0, edgesBeforeMe + above + below - 1)] - os + edgeMask[edgesBeforeMe + max(0, below + above - 1)];

    nodes.isLeaf[leftChildIndex] = FORCE_LEAF || elemsleft < 1;
    nodes.isLeaf[rightChildIndex] = FORCE_LEAF || elemsright < 1;
    
    nodes.contentCount[leftChildIndex] = elemsleft;
    nodes.contentCount[rightChildIndex] = elemsright;
    
    uint ltmp = scannedEdgeMask[edgesBeforeMe];
    nodes.contentStart[leftChildIndex] = ltmp;
    nodes.contentStart[rightChildIndex] = ltmp + elemsleft;
}

template <
    typename T
>
__global__ void postprocess(const T* __restrict raw, T* trans, Primitive nodesContent, uint N)
{
    RETURN_IF_OOB(N);
    uint prim = nodesContent.primIndex[id];
    trans[id] = raw[prim];
}

__global__ void postprocessNodes(Node nodes, uint* dfoMask, uint N)
{
    RETURN_IF_OOB(N);
    uint m = dfoMask[id];

    //nodes.contentCount[m]
}

#if defined DYNAMIC_PARALLELISM

__global__ void startSort(Node nodes, IndexedEdge* edges, uint offset);

__global__ void startGetMinSplits(Node nodes, IndexedSplit* splits, uint offset);

#endif