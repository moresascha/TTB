#pragma once

#include "tree.h"
#include <Nutty.h>
#include "geometry.h"
#include <DeviceBuffer.h>
#include <Copy.h>
#include <Scan.h>
#include <cuda/Stream.h>
#include <cuda/cuda_helper.h>
#include "memory.h"
#include "device_heap.h"

#define EDGE_START ((byte)0)
#define EDGE_END   ((byte)1)

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

    nutty::Iterator<T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
    > Begin(CTbyte index)
    {
        return m_buffer[index].Begin();
    }

    nutty::Iterator<T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
    > End(CTbyte index)
    {
        return m_buffer[index].End();
    }

    void Toggle(void)
    {
        m_current = (m_current + 1) % 2;
    }

    size_t Size(void)
    {
        return m_buffer[0].Size();
    }

    void ZeroMem(void)
    {
        nutty::ZeroMem(m_buffer[0]);
        nutty::ZeroMem(m_buffer[1]);
    }
};

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

    void _DebugDrawNodes(CTuint parent, AABB aabb, ICTTreeDebugLayer* dbLayer) const;

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

struct MakeLeavesResult
{
    CTuint leafPrimitiveCount;
    CTuint leafCount;
    CTuint interiorPrimitiveCount;
};

class cuKDTreeBitonicSearch : public cuKDTree
{
private:
    void ClearBuffer(void);
    void GrowPrimitiveEventMemory(void);
    void GrowNodeMemory(void);
    void GrowPerLevelNodeMemory(CTuint newSize);
    void InitBuffer(void);

    void PrintStatus(const char* msg = NULL);

    void ValidateTree(void);

    void ComputeSAH_Splits(
        CTuint nodeCount, 
        CTuint primitiveCount, 
        const CTuint* hNodesContentCount, 
        const CTuint* nodesContentCount, 
        const CTbyte* isLeaf,
        NodeContent nodesContent);

    CTuint CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint range);

    MakeLeavesResult MakeLeaves(
        nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin,
        CTuint g_nodeOffset, 
        CTuint nodeOffset, 
        CTuint nodeCount, 
        CTuint primitiveCount, 
        CTuint currentLeafCount, 
        CTuint leafContentOffset,
        CTuint initNodeToLeafIndex);

    nutty::DeviceBuffer<CTuint> m_edgeMask;
    nutty::DeviceBuffer<CTuint> m_scannedEdgeMask;
    nutty::DeviceBuffer<CTuint> m_edgeMaskSums;

    nutty::HostBuffer<CTuint> m_hNodesContentCount;

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

    nutty::DeviceBuffer<CTuint> m_lastNodeContentStartAdd;

    nutty::DeviceBuffer<CTuint> m_activeNodesThisLevel;
    nutty::DeviceBuffer<CTuint> m_activeNodes;
    nutty::DeviceBuffer<CTuint> m_newActiveNodes;
    nutty::DeviceBuffer<CTbyte> m_activeNodesIsLeaf;
    nutty::DeviceBuffer<CTbyte> m_activeNodesIsLeafCompacted;

    DoubleBuffer<BBox> m_nodesBBox;

    nutty::DeviceBuffer<CTuint> m_newInteriorContent;
    nutty::DeviceBuffer<CTuint> m_newPrimNodeIndex;
    nutty::DeviceBuffer<CTuint> m_newPrimPrefixSum;
    nutty::DeviceBuffer<CTuint> m_newNodesContentCount;
    nutty::DeviceBuffer<CTuint> m_newNodesContentStartAdd;

    nutty::cuStreamPool m_pool;
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

struct PlaneEvents
{
    nutty::DeviceBuffer<CTreal> x;
    nutty::DeviceBuffer<CTreal> y;
    nutty::DeviceBuffer<CTreal> z;

    nutty::DeviceBuffer<CTreal2> dx;

    void Resize(CTuint size)
    {
        x.Resize(size);
        y.Resize(size);
        z.Resize(size);
        x.Resize(size);
        x.Resize(size);
    }
};

struct cuEventLine
{
    IndexedEvent* indexedEvent;
    CTbyte* type;
    CTuint* nodeIndex;
    CTuint* prefixSum;
    CTuint* primId;
    BBox* ranges;

    const CTuint* __restrict scannedEventTypeStartMask;
    const CTuint* __restrict scannedEventTypeEndMask;
    const CTuint* __restrict eventTypeMaskSumsStart;
    const CTuint* __restrict eventTypeMaskSumsEnd;
};

struct EventLine
{
    DoubleBuffer<IndexedEvent> indexedEvent;
    DoubleBuffer<CTbyte> type;
    DoubleBuffer<CTuint> nodeIndex;
    DoubleBuffer<CTuint> prefixSum;
    DoubleBuffer<CTuint> primId;
    DoubleBuffer<BBox> ranges;

    nutty::DeviceBuffer<CTuint> scannedEventTypeStartMask;
    nutty::DeviceBuffer<CTuint> scannedEventTypeEndMask;
    nutty::DeviceBuffer<CTuint> eventTypeMaskSumsStart;
    nutty::DeviceBuffer<CTuint> eventTypeMaskSumsEnd;

    nutty::DeviceBuffer<CTuint> mask;
    nutty::DeviceBuffer<CTuint> scannedMasks;
    nutty::DeviceBuffer<CTuint> maskSums;

    CTbyte toggleIndex;
    CTuint eventCount;

    EventLine(void) : toggleIndex(0)
    {

    }

    CTuint GetEventCount(void)
    {
        return eventCount;
    }

    void Resize(CTuint size)
    {
        if(indexedEvent.Size() >= size)
        {
            return;
        }
        
        mask.Resize(size);
        scannedMasks.Resize(size);
        maskSums.Resize(size);

        indexedEvent.Resize(size);
        type.Resize(size);
        nodeIndex.Resize(size);
        prefixSum.Resize(size);
        primId.Resize(size);
        ranges.Resize(size);

        scannedEventTypeStartMask.Resize(size);
        scannedEventTypeEndMask.Resize(size);
        eventTypeMaskSumsStart.Resize(size);
        eventTypeMaskSumsEnd.Resize(size);
    }

    size_t Size(void)
    {
        return indexedEvent.Size();
    }

    cuEventLine GetPtr(CTbyte index)
    {
        cuEventLine events;
        events.indexedEvent = indexedEvent.Begin(index)();
        events.type = type.Begin(index)();
        events.nodeIndex = nodeIndex.Begin(index)();
        events.prefixSum = prefixSum.Begin(index)();
        events.primId = primId.Begin(index)();
        events.ranges = ranges.Begin(index)();

        events.scannedEventTypeStartMask = scannedEventTypeStartMask.Begin()();
        events.scannedEventTypeEndMask = scannedEventTypeEndMask.Begin()();
        events.eventTypeMaskSumsStart = eventTypeMaskSumsStart.Begin()();
        events.eventTypeMaskSumsEnd = eventTypeMaskSumsEnd.Begin()();

        return events;
    }

    void ZeroMem(CTuint offset)
    {
//         for(CTbyte i = 0; i < 2; ++i)
//         {
//             CTuint length = indexedEvent[i].Size() - offset;
//             nutty::ZeroMem(indexedEvent[i].Begin() + offset, indexedEvent[i].Begin() + length);
//             nutty::ZeroMem(type[i]);
//             nutty::ZeroMem(nodeIndex[i]);
//             nutty::ZeroMem(prefixSum[i]);
//             nutty::ZeroMem(primId[i]);
//         }
    }

    void ScanEventTypes(void);

    void ScanEvents(CTuint length);

    void CompactClippedEvents(CTuint length);

    void Toogle(void)
    {
        toggleIndex ^= 1;
    }

    cuEventLine GetSrc(void)
    {
        return GetPtr(toggleIndex);
    }

    cuEventLine GetDst(void)
    {
        return GetPtr((toggleIndex+1)%2);
    }
};

typedef struct cuEventLineTriple
{
    cuEventLine lines[3];

    __device__ cuEventLineTriple(void)
    {

    }

    __host__ cuEventLineTriple(EventLine line[3])
    {
        lines[0] = line[0].GetDst();
        lines[1] = line[1].GetDst();
        lines[2] = line[2].GetDst();
    }

    __host__ cuEventLineTriple(EventLine line[3], CTbyte index)
    {
        lines[0] = line[0].GetPtr(index);
        lines[1] = line[1].GetPtr(index);
        lines[2] = line[2].GetPtr(index);
    }

    __device__ cuEventLine& getLine(CTbyte index)
    {
        return lines[index];
    }

} cuEventLineTriple;

class cuKDTreeScan : public cuKDTree
{
private:
    void ClearBuffer(void);
    void GrowPrimitiveEventMemory(void);
    void GrowNodeMemory(void);
    void GrowPerLevelNodeMemory(CTuint newSize);
    void InitBuffer(void);

    void PrintStatus(const char* msg = NULL);

    void ValidateTree(void);

    void ComputeSAH_Splits(
        CTuint nodeCount, 
        CTuint primitiveCount, 
        const CTuint* hNodesContentCount, 
        const CTuint* nodesContentCount);

    CTuint CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint range);

    MakeLeavesResult MakeLeaves(
        nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin,
        CTuint g_nodeOffset, 
        CTuint nodeOffset, 
        CTuint nodeCount, 
        CTuint primitiveCount, 
        CTuint currentLeafCount, 
        CTuint leafContentOffset,
        CTuint initNodeToLeafIndex);

    nutty::DeviceBuffer<CTuint> m_eventMask;
    nutty::DeviceBuffer<CTuint> m_scannedeventMask;
    nutty::DeviceBuffer<CTuint> m_eventMaskSums;

    nutty::HostBuffer<CTuint> m_hNodesContentCount;

    Split m_splits;
    nutty::DeviceBuffer<IndexedSAHSplit> m_splits_IndexedSplit;
    nutty::DeviceBuffer<CTreal> m_splits_Plane;
    nutty::DeviceBuffer<CTbyte> m_splits_Axis;
    nutty::DeviceBuffer<CTuint> m_splits_Above;
    nutty::DeviceBuffer<CTuint> m_splits_Below;

    EventLine m_events3[3];

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

    nutty::DeviceBuffer<CTuint> m_lastNodeContentStartAdd;

    nutty::DeviceBuffer<CTuint> m_activeNodesThisLevel;
    nutty::DeviceBuffer<CTuint> m_activeNodes;
    nutty::DeviceBuffer<CTuint> m_newActiveNodes;
    nutty::DeviceBuffer<CTbyte> m_activeNodesIsLeaf;
    nutty::DeviceBuffer<CTbyte> m_activeNodesIsLeafCompacted;

    DoubleBuffer<BBox> m_nodesBBox;

    nutty::DeviceBuffer<CTuint> m_newInteriorContent;
    nutty::DeviceBuffer<CTuint> m_newPrimNodeIndex;
    nutty::DeviceBuffer<CTuint> m_newPrimPrefixSum;
    nutty::DeviceBuffer<CTuint> m_newNodesContentCount;
    nutty::DeviceBuffer<CTuint> m_newNodesContentStartAdd;

    nutty::cuStreamPool m_pool;
public:
    cuKDTreeScan(void)
    {
    }

    CT_RESULT Update(void);

    ~cuKDTreeScan(void)
    {

    }

    add_uuid_header(cuKDTreeScan);
};