#pragma once

#include "tree.h"
#include <Nutty.h>
#include "geometry.h"
#include <DeviceBuffer.h>
#include <cuda/cuda_helper.h>
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
    CTbyte m_current;

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

    nutty::DeviceBuffer<T>& Get(CTbyte index)
    {
        assert(index < 2);
        return m_buffer[index];
    }

    nutty::DeviceBuffer<T>& operator[](CTbyte index)
    {
        return Get(index);
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

__forceinline __device__ __host__ CTuint elemsOnLevel(byte l)
{
    return (1 << l);
}

__forceinline __device__ __host__ CTuint elemsBeforeNextLevel(byte l)
{
    return elemsBeforeLevel(l + 1);
}

struct Primitive
{
    CTuint* primIndex;
    CTuint* nodeIndex;
    CTuint* prefixSum;
};

struct IndexedEvent
{
    CTuint index;
    CTreal v;
};

struct Edge
{
    CTbyte* type;
    IndexedEvent* indexedEdge;
    CTuint* nodeIndex;
    CTuint* primId;
    CTuint* prefixSum;
};

struct IndexedSAHSplit
{
    CTreal sah;
    CTuint index;
};

struct Split
{
    IndexedSAHSplit* indexedSplit;
    CTbyte* axis;
    CTuint* below;
    CTuint* above;
    CTreal* v;
};

struct Node
{
    BBox* aabb;
    CTbyte* isLeaf;
    CTreal* split;
    CTbyte* splitAxis;
    CTuint* contentCount;
    CTuint* contentStart;
    CTuint* content;
    CTuint* above;
    CTuint* below;
};

struct EdgeSort
{
    __device__ char operator()(IndexedEvent t0, IndexedEvent t1)
    {
        return t0.v > t1.v;
    }
};

struct ReduceIndexedSplit
{
    __device__ __host__ IndexedSAHSplit operator()(IndexedSAHSplit t0, IndexedSAHSplit t1)
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
protected:
    ICTTreeNode* m_node;
    CT_GEOMETRY_TOPOLOGY m_topo;
    CTuint m_interiorNodesCount;
    CTuint m_leafNodesCount;
    CTbyte m_depth;
    CTbyte m_maxDepth;
    CTuint m_flags;
    CTbool m_initialized;
    AABB m_sceneAABB;

    nutty::DeviceBuffer<CTreal3> m_orginalVertices;
    nutty::DeviceBuffer<CTreal3> m_currentTransformedVertices;

    nutty::DeviceBuffer<CTreal3> m_kdPrimitives;

    nutty::DeviceBuffer<CTuint> m_edgeMask;
    nutty::DeviceBuffer<CTuint> m_scannedEdgeMask;
    nutty::DeviceBuffer<CTuint> m_edgeMaskSums;

    nutty::DeviceBuffer<CTreal3> m_bboxMin;
    nutty::DeviceBuffer<CTreal3> m_bboxMax;
    nutty::DeviceBuffer<BBox> m_sceneBBox;

    nutty::HostBuffer<CTuint> m_hNodesContentCount;

    Node m_nodes;
    nutty::DeviceBuffer<BBox> m_nodesBBox;
    nutty::DeviceBuffer<CTbyte> m_nodesIsLeaf;
    nutty::DeviceBuffer<CTbyte> m_nodesSplitAxis;
    nutty::DeviceBuffer<CTreal> m_nodesSplit;
    nutty::DeviceBuffer<CTuint> m_nodesContentCount;
    nutty::DeviceBuffer<CTuint> m_nodesStartAdd;

    nutty::DeviceBuffer<CTuint> m_nodesAbove;
    nutty::DeviceBuffer<CTuint> m_nodesBelow;

    Split m_splits;
    nutty::DeviceBuffer<IndexedSAHSplit> m_splitsIndexedSplit;
    nutty::DeviceBuffer<CTreal> m_splitsSplit;
    nutty::DeviceBuffer<CTbyte> m_splitsAxis;
    nutty::DeviceBuffer<CTuint> m_splitsAbove;
    nutty::DeviceBuffer<CTuint> m_splitsBelow;

    Edge m_edges[2];
    nutty::DeviceBuffer<IndexedEvent> m_edgesIndexedEdge;
    DoubleBuffer<CTbyte> m_edgesType;
    DoubleBuffer<CTuint> m_edgesNodeIndex;
    DoubleBuffer<CTuint> m_edgesPrimId;
    DoubleBuffer<CTuint> m_edgesPrefixSum;
    nutty::DeviceBuffer<CTuint> m_scannedEdgeTypeStartMask;
    nutty::DeviceBuffer<CTuint> m_scannedEdgeTypeEndMask;
    nutty::DeviceBuffer<CTuint> m_edgeTypeMaskSums;

    Primitive m_nodesContent;
    nutty::DeviceBuffer<CTuint> m_primIndex;
    nutty::DeviceBuffer<CTuint> m_primNodeIndex;
    nutty::DeviceBuffer<CTuint> m_primPrefixSum;

    nutty::DeviceBuffer<BBox> m_primAABBs;
    nutty::DeviceBuffer<BBox> m_tPrimAABBs;

    std::vector<CTulong> m_linearGeoHandles;
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

    virtual CT_RESULT Init(CTuint flags);

    virtual CT_RESULT Update(void) = 0;

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
        return (CTuint)m_orginalVertices.Size() / 3;
    }

    CT_RESULT Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData = NULL);

    void TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix);

    const void* GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const;

    CT_RESULT AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle);

    const ICTAABB* GetAxisAlignedBB(void) const
    {
        return &m_sceneAABB;
    }

    virtual ~cuKDTree(void);

    add_uuid_header(cuKDTree);
};

class cuKDTreeBitonicSearch : public cuKDTree
{
private:
    void ClearBuffer(void);
    void GrowMemory(void);
    void InitBuffer(void);

public:
    cuKDTreeBitonicSearch(void)
    {

    }

    CT_RESULT Update(void);

    ~cuKDTreeBitonicSearch(void)
    {

    }

    add_uuid_header(cuKDTreeBitonicSearch);
};