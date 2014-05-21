#pragma once
#include "tree.h"
#include <Nutty.h>
#include "geometry.h"
#include <DeviceBuffer.h>
#include "memory.h"
#include "shared_types.h"

#undef DYNAMIC_PARALLELISM

#ifdef DYNAMIC_PARALLELISM
#pragma comment(lib, "cudadevrt.lib")
#endif

#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)

#define EDGE_START ((byte)0)
#define EDGE_END   ((byte)1)

#define INVALID_ADDRESS ((uint)-1)
#define FLT_MAX 3.402823466e+38F
#define FLT_MAX_DIV2 (FLT_MAX/2.0f)

template <
    typename T
>
class DoubleBuffer
{
private:
    nutty::DeviceBuffer<T> m_buffer[2];
    byte m_current;

public:
    DoubleBuffer(void) :  m_current(0)
    {

    }

    void Resize(size_t size)
    {
        for(byte i = 0; i < 2; ++i)
        {
            m_buffer[i].Resize(size);
        }
    }

    nutty::DeviceBuffer<T>& Get(byte index)
    {
        assert(index < 2);
        return m_buffer[index];
    }

    nutty::DeviceBuffer<T>& GetCurrent(void)
    {
        return m_buffer[m_current];
    }

    void Toggle(void)
    {
        m_current = (m_current + 1) % 2;
    }

    void ZeroMem(void)
    {
        nutty::ZeroMem(m_buffer[0]);
        nutty::ZeroMem(m_buffer[1]);
    }
};

__forceinline __device__ __host__ uint elemsBeforeLevel(byte l)
{
    return (1 << l) - 1;
}

__forceinline __device__ __host__ uint elemsOnLevel(byte l)
{
    return (1 << l);
}

__forceinline __device__ __host__ uint elemsBeforeNextLevel(byte l)
{
    return elemsBeforeLevel(l + 1);
}

struct Primitive
{
    uint* primIndex;
    uint* nodeIndex;
    uint* prefixSum;
};

struct IndexedEdge
{
    uint index;
    float v;
};

struct Edge
{
    byte* type;
    IndexedEdge* indexedEdge;
    uint* nodeIndex;
    uint* primId;
    uint* prefixSum;
};

struct IndexedSplit
{
    float sah;
    uint index;
};

struct Split
{
    IndexedSplit* indexedSplit;
    byte* axis;
    uint* below;
    uint* above;
    float* v;
};

struct Node
{
    BBox* aabb;
    byte* isLeaf;
    float* split;
    byte* splitAxis;
    uint* contentCount;
    uint* contentStart;
    uint* content;
    uint* above;
    uint* below;
};

struct EdgeSort
{
    __device__ char operator()(IndexedEdge t0, IndexedEdge t1)
    {
        return t0.v > t1.v;
    }
};

struct ReduceIndexedSplit
{
    __device__ __host__ IndexedSplit operator()(IndexedSplit t0, IndexedSplit t1)
    {
        return t0.sah <= t1.sah ? t0 : t1;
    }
};

// struct ReadOnlyNode
// {
//     const BBox* __restrict aabb;
//     const byte* __restrict isLeaf;
//     const float* __restrict split;
//     const byte* __restrict splitAxis;
//     const uint* __restrict contentCount;
//     const uint* __restrict contentStart;
//     const uint* __restrict content;
//     const uint* __restrict above;
//     const uint* __restrict below;
// };

class cuKDTree : public ICTTree
{
private:
    ICTTreeNode* m_node;
    CT_GEOMETRY_TOPOLOGY m_topo;
    CTuint m_interiorNodesCount;
    CTuint m_leafNodesCount;
    byte m_depth;
    byte m_maxDepth;
    CTuint m_flags;
    CTbool m_initialized;
    AABB m_sceneAABB;

    nutty::DeviceBuffer<uint> m_depthFirstMask;

    nutty::DeviceBuffer<CTreal3> m_primitives;
    nutty::DeviceBuffer<CTreal3> m_tprimitives;

    nutty::DeviceBuffer<uint> m_edgeMask;
    nutty::DeviceBuffer<uint> m_scannedEdgeMask;
    nutty::DeviceBuffer<uint> m_edgeMaskSums;

    nutty::DeviceBuffer<float3> m_bboxMin;
    nutty::DeviceBuffer<float3> m_bboxMax;
    nutty::DeviceBuffer<BBox> m_sceneBBox;

    nutty::HostBuffer<uint> m_hNodesContentCount;

    Node m_nodes;
    nutty::DeviceBuffer<BBox> m_nodesBBox;
    nutty::DeviceBuffer<byte> m_nodesIsLeaf;
    nutty::DeviceBuffer<byte> m_nodesSplitAxis;
    nutty::DeviceBuffer<float> m_nodesSplit;
    nutty::DeviceBuffer<uint> m_nodesContentCount;
    nutty::DeviceBuffer<uint> m_nodesStartAdd;
    nutty::DeviceBuffer<uint> m_nodesAbove;
    nutty::DeviceBuffer<uint> m_nodesBelow;

    Node m_dfoNodes;
    nutty::DeviceBuffer<byte> m_dfoNodesIsLeaf;
    nutty::DeviceBuffer<byte> m_dfoNodesSplitAxis;
    nutty::DeviceBuffer<float> m_dfoNodesSplit;
    nutty::DeviceBuffer<uint> m_dfoNodesContentCount;
    nutty::DeviceBuffer<uint> m_dfoNodesStartAdd;

    Split m_splits;
    nutty::DeviceBuffer<IndexedSplit> m_splitsIndexedSplit;
    nutty::DeviceBuffer<float> m_splitsSplit;
    nutty::DeviceBuffer<byte> m_splitsAxis;
    nutty::DeviceBuffer<uint> m_splitsAbove;
    nutty::DeviceBuffer<uint> m_splitsBelow;

    Edge m_edges[2];
    nutty::DeviceBuffer<IndexedEdge> m_edgesIndexedEdge;
    DoubleBuffer<byte> m_edgesType;
    DoubleBuffer<uint> m_edgesNodeIndex;
    DoubleBuffer<uint> m_edgesPrimId;
    DoubleBuffer<uint> m_edgesPrefixSum;

    Primitive m_nodesContent;
    nutty::DeviceBuffer<uint> m_primIndex;
    nutty::DeviceBuffer<uint> m_primNodeIndex;
    nutty::DeviceBuffer<uint> m_primPrefixSum;

    nutty::DeviceBuffer<BBox> m_primAABBs;
    nutty::DeviceBuffer<BBox> m_tPrimAABBs;

    std::map<CTGeometryHandle, GeometryRange> m_handleRangeMap;

public:
    cuKDTree(void);

    void OnGeometryMoved(const CTGeometryHandle geo)
    {

    }

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo)
    {
        m_topo = topo;
        return CT_SUCCESS;
    }

    CT_RESULT Init(CTuint flags);

    CT_RESULT Update(void);

    ICTTreeNode* GetRoot(void) const
    {
        return m_node;
    }

    CT_RESULT DebugDraw(ICTTreeDebugLayer* dbLayer) const;

    CT_RESULT AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle);

    void SetDepth(CTbyte depth);

    CTuint GetDepth(void) const
    {
        return m_depth;
    }

    CTuint GetInteriorNodesCount(void) const
    {
        return m_interiorNodesCount;
    }
    
    CTuint GetLeafNodesCount(void) const
    {
        return m_leafNodesCount;
    }
    
    CT_TREE_DEVICE GetDeviceType(void) const
    {
        return eCT_GPU;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return m_topo;
    }

    const CTGeometryHandle* GetGeometry(CTuint* gc);

    const CTreal* GetRawPrimitives(CTuint* bytes) const;

    CTuint GetPrimitiveCount(void) const
    {
        return (CTuint)m_primitives.Size() / 3;
    }

    CT_RESULT Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData = NULL);

    void TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix);

    const void* GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const;

    CT_RESULT AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle);

    const ICTAABB* GetAxisAlignedBB(void) const
    {
        return &m_sceneAABB;
    }

    ~cuKDTree(void);

    add_uuid_header(cuKDTree);
};