#pragma once

#include "tree.h"
#include <Nutty.h>
#include "geometry.h"
#include <DeviceBuffer.h>
#include <Copy.h>
#include <Scan.h>
#include <cuda/cuda_helper.h>
#include "memory.h"
#include "shared_types.h"
#include "device_heap.h"

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

struct NodeContent
{
    CTuint* primIndex;
    CTuint* nodeIndex;
    CTuint* prefixSum;
    CTbyte* primIsLeaf;
};

struct IndexedEvent
{
    CTuint index;
    CTreal v;
};

struct Event
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

    __device__  __host__ IndexedSAHSplit(void)
    {

    }
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
    CTbyte* isLeaf;
    CTreal* split;
    CTbyte* splitAxis;
    CTuint* contentCount;
    CTuint* contentStart;
    CTuint* leftChild;
    CTuint* rightChild;
    CTuint* nodeToLeafIndex;
};

struct EventSort
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
        return t0.sah < t1.sah ? t0 : t1;
    }
};

class cuKDTree : public ICTTree
{
protected:
    ICTTreeNode* m_node;
    CT_GEOMETRY_TOPOLOGY m_topo;
    CTuint m_interiorNodesCount;
    CTuint m_leafNodesCount;
    CTbyte m_depth;
    CTuint m_flags;
    CTbool m_initialized;
    AABB m_sceneAABB;

    nutty::DeviceBuffer<CTreal3> m_orginalVertices;
    nutty::DeviceBuffer<CTreal3> m_currentTransformedVertices;

    nutty::DeviceBuffer<CTreal3> m_kdPrimitives;

    nutty::DeviceBuffer<CTreal3> m_bboxMin;
    nutty::DeviceBuffer<CTreal3> m_bboxMax;
    nutty::DeviceBuffer<BBox> m_sceneBBox;

    nutty::DeviceBuffer<BBox> m_primAABBs;
    nutty::DeviceBuffer<BBox> m_tPrimAABBs;
    nutty::DeviceBuffer<BBox> m_linearPrimAABBs; //for debugging

    Node m_nodes;
    nutty::DeviceBuffer<CTbyte> m_nodes_IsLeaf;
    nutty::DeviceBuffer<CTbyte> m_nodes_SplitAxis;
    nutty::DeviceBuffer<CTreal> m_nodes_Split;
    nutty::DeviceBuffer<CTuint> m_nodes_ContentCount;
    nutty::DeviceBuffer<CTuint> m_nodes_ContentStartAdd;

    nutty::DeviceBuffer<CTuint> m_nodes_LeftChild;
    nutty::DeviceBuffer<CTuint> m_nodes_RightChild;
    nutty::DeviceBuffer<CTuint> m_nodes_NodeIdToLeafIndex;

    nutty::DeviceBuffer<CTuint> m_leafNodesContent;
    nutty::DeviceBuffer<CTuint> m_leafNodesContentCount;
    nutty::DeviceBuffer<CTuint> m_leafNodesContentStart;

    std::vector<CTulong> m_linearGeoHandles;
    std::map<CTGeometryHandle, GeometryRange> m_handleRangeMap;

    void _DebugDrawNodes(CTuint parent, ICTTreeDebugLayer* dbLayer) const;

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

class Scanner
{
private:
    nutty::DeviceBuffer<CTuint> m_scannedData;
    nutty::DeviceBuffer<CTuint> m_sums;

public:
    void Resize(CTuint size)
    {
        m_scannedData.Resize(size);
        m_sums.Resize(size);
    }

    template<
        typename Iterator,
        typename Operator
    >
    void IncScan(Iterator& begin, Iterator& end, Operator op)
    {
        nutty::InclusiveScan(begin, end, m_scannedData.Begin(), m_sums.Begin(), op);
    }

    template<
        typename Iterator,
        typename Operator
    >
    void ExcScan(Iterator& begin, Iterator& end, Operator op)
    {
        nutty::ExclusiveScan(begin, end, m_scannedData.Begin(), m_sums.Begin(), op);
    }

    const nutty::DeviceBuffer<CTuint>& GetPrefixSum(void)
    {
        return m_scannedData;
    }
};

class cuKDTreeBitonicSearch : public cuKDTree
{
private:
    void ClearBuffer(void);
    void GrowMemory(void);
    void GrowNodeMemory(void);
    void GrowPerLevelNodeMemory(void);
    void InitBuffer(void);

    void ComputeSAH_Splits(
        CTuint nodeCount, 
        CTuint nodeOffset,
        CTuint primitiveCount, 
        const CTuint* hNodesContentCount, 
        const CTuint* nodesContentCount, 
        const CTbyte* isLeaf,
        NodeContent nodesContent);

    CTuint GetLeavesOnLevel(CTuint nodeOffset, CTuint nodeCount);

    nutty::DeviceBuffer<CTuint> m_edgeMask;
    nutty::DeviceBuffer<CTuint> m_scannedEdgeMask;
    nutty::DeviceBuffer<CTuint> m_edgeMaskSums;

    nutty::HostBuffer<CTuint> m_hNodesContentCount;
    nutty::HostBuffer<CTbyte> m_hIsLeaf;

    Split m_splits;
    nutty::DeviceBuffer<IndexedSAHSplit> m_splits_IndexedSplit;
    nutty::DeviceBuffer<CTreal> m_splits_Plane;
    nutty::DeviceBuffer<CTbyte> m_splits_Axis;
    nutty::DeviceBuffer<CTuint> m_splits_Above;
    nutty::DeviceBuffer<CTuint> m_splits_Below;

    Event m_events[2];
    nutty::DeviceBuffer<IndexedEvent> m_events_IndexedEdge;
    DoubleBuffer<CTbyte> m_events_Type;
    DoubleBuffer<CTuint> m_events_NodeIndex;
    DoubleBuffer<CTuint> m_events_PrimId;
    DoubleBuffer<CTuint> m_events_PrefixSum;

    nutty::DeviceBuffer<CTuint> m_scannedEventTypeStartMask;
    nutty::DeviceBuffer<CTuint> m_scannedEventTypeEndMask;
    nutty::DeviceBuffer<CTuint> m_eventTypeMaskSums;

    NodeContent m_nodesContent;
    nutty::DeviceBuffer<CTuint> m_primIndex;
    nutty::DeviceBuffer<CTuint> m_primNodeIndex;
    nutty::DeviceBuffer<CTuint> m_primPrefixSum;
    nutty::DeviceBuffer<CTbyte> m_primIsLeaf;

    Scanner m_primIsLeafScanner;
    Scanner m_primIsNoLeafScanner;
    Scanner m_leafCountScanner;
    Scanner m_interiorCountScanner;
    Scanner m_interiorContentScanner;
    Scanner m_leafContentScanner;

    nutty::DeviceBuffer<CTuint> m_maskedInteriorContent;
    nutty::DeviceBuffer<CTuint> m_maskedleafContent;

    DoubleBuffer<BBox> m_nodesBBox;

    nutty::DeviceBuffer<CTuint> m_newInteriorContent;
    nutty::DeviceBuffer<CTuint> m_newPrimNodeIndex;
    nutty::DeviceBuffer<CTuint> m_newPrimPrefixSum;
    nutty::DeviceBuffer<CTuint> m_newNodesContentCount;
    nutty::DeviceBuffer<CTuint> m_newNodesContentStartAdd;

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

class cudpKDTree : public cuKDTree
{
private:
    void ClearBuffer(void);
    void InitBuffer(void);
    cuDeviceHeap m_deviceHeap;

public:
    cudpKDTree(void) { }

    CT_RESULT Update(void) { return CT_NOT_YET_IMPLEMENTED; }

    ~cudpKDTree(void)
    {

    }

    add_uuid_header(cudpKDTree);
};