
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

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

    nutty::Scanner m_primIsLeafScanner;
    nutty::Scanner m_primIsNoLeafScanner;
    nutty::Scanner m_leafCountScanner;
    nutty::Scanner m_interiorCountScanner;
    nutty::Scanner m_interiorContentScanner;
    nutty::Scanner m_leafContentScanner;

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

#include <chimera/Timer.h>
#include "cuKDTree.h"
#include "kd_kernel.h"
#include "shared_kernel.h"
#include "shared_types.h"
#include <Reduce.h>
#include <Sort.h>
#include <Scan.h>
#include <queue>
#include <ForEach.h>
#include <Fill.h>
#include <Functions.h>
#include <cuda/Globals.cuh>
#include "buffer_print.h"

#undef PRINT_OUT
#ifndef _DEBUG
#undef PRINT_OUT
#endif

#ifndef PRINT_OUT
#undef PRINT_BUFFER(_name)
#undef PRINT_BUFFER_N(_name)
#undef ct_printf

#define PRINT_BUFFER(_name)
#define PRINT_BUFFER_N(_name)
#define ct_printf(...)
#endif

struct ShrinkContentStartAdd
{
    CTuint leafCount;
    __device__ __host__ ShrinkContentStartAdd(CTuint lc) : leafCount(lc)
    {

    }

    __device__ __host__ ShrinkContentStartAdd(const ShrinkContentStartAdd& s) : leafCount(s.leafCount)
    {

    }

    __device__ CTuint operator()(CTuint elem)
    {
        if(elem < leafCount)
        {
            return elem;
        }
        return elem - leafCount;
    }
};

template<>
struct ShrdMemory<IndexedSAHSplit>
{
    __device__ IndexedSAHSplit* Ptr(void) 
    { 
        extern __device__ __shared__ IndexedSAHSplit s_split[];
        return s_split;
    }
};

template<>
struct ShrdMemory<BBox>
{
    __device__ BBox* Ptr(void) 
    { 
        extern __device__ __shared__ BBox s_bbox[];
        return s_bbox;
    }
};

template<>
struct ShrdMemory<IndexedEvent>
{
    __device__ IndexedEvent* Ptr(void) 
    { 
        extern __device__ __shared__ IndexedEvent s_edge[];
        return s_edge;
    }
};

struct AxisSort
{
    char axis;
    AxisSort(char a) : axis(a)
    {

    }
    __device__ __host__ char operator()(const CTreal3& f0, const CTreal3& f1)
    {
        return getAxis(f0, axis) > getAxis(f1, axis);
    }
};

struct float3min
{
    __device__ float3 operator()(const float3& t0, const float3& t1)
    {
        float3 r;
        r.x = nutty::binary::Min<float>()(t0.x, t1.x);
        r.y = nutty::binary::Min<float>()(t0.y, t1.y);
        r.z = nutty::binary::Min<float>()(t0.z, t1.z);
        return r;
    }
};

struct float3max
{
    __device__  float3 operator()(const float3& t0, const float3& t1)
    {
        float3 r;
        r.x = nutty::binary::Max<float>()(t0.x, t1.x);
        r.y = nutty::binary::Max<float>()(t0.y, t1.y);
        r.z = nutty::binary::Max<float>()(t0.z, t1.z);
        return r;
    }
};

struct ReduceBBox
{
    __device__  BBox operator()(const BBox& t0, const BBox& t1)
    {
        BBox bbox;
        bbox.m_min = fminf(t0.m_min, t1.m_min);
        bbox.m_max = fmaxf(t0.m_max, t1.m_max);
        return bbox;
    }
};

template <
    typename T
>
struct InvTypeOp
{
    __device__ T operator()(T elem)
    {
        return (elem < 2) * (elem ^ 1);
    }

    T GetNeutral(void)
    {
        return 0;
    }
};

template <
    typename T
>
struct TypeOp
{
    __device__ T operator()(T elem)
    {
        return (elem < 2) * elem;
    }

    T GetNeutral(void)
    {
        return 0;
    }
};

template <
    typename T
>
struct EventStartScanOp
{
    __device__ T operator()(T elem)
    {
        return elem ^ 1;
    }

    T GetNeutral(void)
    {
        return 1;
    }
};

template <
    typename T
>
struct EventEndScanOp
{
    __device__ T operator()(T elem)
    {
        return elem;
    }

    T GetNeutral(void)
    {
        return 0;
    }
};

void cuKDTreeBitonicSearch::InitBuffer(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;

    m_depth = (byte)min(64, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));

    m_primAABBs.Resize(primitiveCount); nutty::ZeroMem(m_primAABBs);

    GrowNodeMemory();
    GrowPerLevelNodeMemory(64);
    GrowPrimitiveEventMemory();

    ClearBuffer();
}

void cuKDTreeBitonicSearch::ClearBuffer(void)
{
    nutty::ZeroMem(m_edgeMask);
    nutty::ZeroMem(m_scannedEdgeMask);
    nutty::ZeroMem(m_edgeMaskSums);

    nutty::ZeroMem(m_nodesBBox[0]);
    nutty::ZeroMem(m_nodesBBox[1]);

    nutty::ZeroMem(m_nodes_ContentCount);
    nutty::ZeroMem(m_nodes_IsLeaf);
    nutty::ZeroMem(m_nodes_Split);
    nutty::ZeroMem(m_nodes_ContentStartAdd);
    nutty::ZeroMem(m_nodes_SplitAxis);
    nutty::ZeroMem(m_nodes_LeftChild);
    nutty::ZeroMem(m_nodes_RightChild);

    nutty::ZeroMem(m_splits_Above);
    nutty::ZeroMem(m_splits_Below);
    nutty::ZeroMem(m_splits_Axis);
    nutty::ZeroMem(m_splits_Plane);

    nutty::ZeroMem(m_leafNodesContentCount);
    nutty::ZeroMem(m_leafNodesContentStart);

    m_events_NodeIndex.ZeroMem();
    m_events_PrimId.ZeroMem();
    m_events_Type.ZeroMem();
    m_events_PrefixSum.ZeroMem();
}

void cuKDTreeBitonicSearch::GrowPerLevelNodeMemory(CTuint newSize)
{
    m_activeNodesIsLeaf.Resize(newSize);
    m_activeNodes.Resize(newSize);
    m_activeNodesThisLevel.Resize(newSize);
    m_newActiveNodes.Resize(newSize);
    m_nodesBBox.Resize(newSize);
    m_nodes_ContentStartAdd.Resize(newSize);
    m_nodes_ContentCount.Resize(newSize);

    m_nodes.isLeaf = m_nodes_IsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodes_SplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodes_Split.GetDevicePtr()();
    m_nodes.contentStart = m_nodes_ContentStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodes_ContentCount.GetDevicePtr()();
    m_nodes.leftChild = m_nodes_LeftChild.GetDevicePtr()();
    m_nodes.rightChild = m_nodes_RightChild.GetDevicePtr()();
    m_nodes.nodeToLeafIndex = m_nodes_NodeIdToLeafIndex.GetDevicePtr()();
}

void cuKDTreeBitonicSearch::GrowNodeMemory(void)
{
    size_t newSize = m_nodes_IsLeaf.Size() ? m_nodes_IsLeaf.Size() * 4 : 32;

    m_nodes_IsLeaf.Resize(newSize);
    m_nodes_Split.Resize(newSize);
    m_nodes_NodeIdToLeafIndex.Resize(newSize);
    m_nodes_SplitAxis.Resize(newSize);
    m_nodes_LeftChild.Resize(newSize);
    m_nodes_RightChild.Resize(newSize);

    m_nodes.isLeaf = m_nodes_IsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodes_SplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodes_Split.GetDevicePtr()();
    m_nodes.contentStart = m_nodes_ContentStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodes_ContentCount.GetDevicePtr()();
    m_nodes.leftChild = m_nodes_LeftChild.GetDevicePtr()();
    m_nodes.rightChild = m_nodes_RightChild.GetDevicePtr()();
    m_nodes.nodeToLeafIndex = m_nodes_NodeIdToLeafIndex.GetDevicePtr()();
}

void cuKDTreeBitonicSearch::GrowPrimitiveEventMemory(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;
    CTuint eventCount = m_primIndex.Size() ? m_primIndex.Size() * 2 : 4 * primitiveCount; //4 times as big
    m_primIndex.Resize(eventCount);
    m_primNodeIndex.Resize(eventCount);
    m_primPrefixSum.Resize(eventCount);
    m_primIsLeaf.Resize(eventCount);

    m_nodesContent.primIndex = m_primIndex.GetDevicePtr()();
    m_nodesContent.nodeIndex = m_primNodeIndex.GetDevicePtr()();
    m_nodesContent.prefixSum = m_primPrefixSum.GetDevicePtr()();
    m_nodesContent.primIsLeaf = m_primIsLeaf.GetDevicePtr()();

    m_splits_Above.Resize(eventCount);
    m_splits_Below.Resize(eventCount);
    m_splits_Axis.Resize(eventCount);
    m_splits_Plane.Resize(eventCount);
    m_splits_IndexedSplit.Resize(eventCount);
    
    m_splits.above = m_splits_Above.GetDevicePtr()();
    m_splits.below = m_splits_Below.GetDevicePtr()();
    m_splits.axis = m_splits_Axis.GetDevicePtr()();
    m_splits.indexedSplit = m_splits_IndexedSplit.GetDevicePtr()();
    m_splits.v = m_splits_Plane.GetDevicePtr()();

    m_events_IndexedEdge.Resize(eventCount);
    m_events_NodeIndex.Resize(eventCount);
    m_events_PrimId.Resize(eventCount);
    m_events_Type.Resize(eventCount);
    m_events_PrefixSum.Resize(eventCount);

    m_events[0].indexedEdge = m_events_IndexedEdge.GetDevicePtr()();
    m_events[0].nodeIndex = m_events_NodeIndex.Get(0).GetDevicePtr()();
    m_events[0].primId = m_events_PrimId.Get(0).GetDevicePtr()();
    m_events[0].type = m_events_Type.Get(0).GetDevicePtr()();
    m_events[0].prefixSum = m_events_PrefixSum.Get(0).GetDevicePtr()();

    m_events[1].indexedEdge = m_events_IndexedEdge.GetDevicePtr()();
    m_events[1].nodeIndex = m_events_NodeIndex.Get(1).GetDevicePtr()();
    m_events[1].primId = m_events_PrimId.Get(1).GetDevicePtr()();
    m_events[1].type = m_events_Type.Get(1).GetDevicePtr()();
    m_events[1].prefixSum = m_events_PrefixSum.Get(1).GetDevicePtr()();

    m_edgeMask.Resize(eventCount);
    m_scannedEdgeMask.Resize(eventCount);
    m_edgeMaskSums.Resize(eventCount); //way to big but /care
}

void cuKDTreeBitonicSearch::PrintStatus(const char* msg /* = NULL */)
{
    ct_printf("PrintStatus: %s\n", msg == NULL ? "" : msg);
    //PRINT_BUFFER(m_activeNodes);
    //PRINT_BUFFER(m_nodesBBox[0]);
    PRINT_BUFFER(m_primNodeIndex);
    //PRINT_BUFFER(m_primIndex);
    //PRINT_BUFFER(m_primPrefixSum);
    PRINT_BUFFER(m_nodes_ContentCount);
    PRINT_BUFFER(m_nodes_ContentStartAdd);
    //PRINT_BUFFER(m_nodes_Split);
    //PRINT_BUFFER(m_nodes_SplitAxis);
    //PRINT_BUFFER(m_nodes_IsLeaf);
    //PRINT_BUFFER(m_leafNodesContent);
    //PRINT_BUFFER(m_leafNodesContentCount);
    //PRINT_BUFFER(m_leafNodesContentStart);
}

void cuKDTreeBitonicSearch::ComputeSAH_Splits(
    CTuint nodeCount,
    CTuint primitiveCount, 
    const CTuint* hNodesContentCount, 
    const CTuint* nodesContentCount, 
    const CTbyte* isLeaf,
    NodeContent nodesContent)
{
    CTuint elementBlock = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);

    //nutty::ZeroMem(m_events_NodeIndex[0]);

    CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
    CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

    CTuint eventCount = 2 * primitiveCount;
    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    Event eventsSrc = m_events[0];
    Event eventsDst = m_events[1];

    createEvents<<<elementGrid, elementBlock>>>(eventsSrc, m_primAABBs.Begin()(), m_nodesBBox[0].Begin()(), nodesContent, primitiveCount);

    DEVICE_SYNC_CHECK();
    m_pool.Reset();
    CTuint start = 0;
   
    chimera::util::HTimer timer;
    cudaDeviceSynchronize();
    timer.Start();

    for(int i = 0; i < nodeCount; ++i)
    {
        CTuint cc = hNodesContentCount[i];
        CTuint length = 2 * cc;
#ifdef _DEBUG
        if(cc <= MAX_ELEMENTS_PER_LEAF)
        {
            assert(0 && "cc <= MAX_ELEMENTS_PER_LEAF");
            //start += length;
            continue;
        }
#endif
        nutty::cuStream& stream = m_pool.PeekNextStream();
        nutty::SetStream(stream);

        nutty::Sort(m_events_IndexedEdge.Begin() + start, m_events_IndexedEdge.Begin() + start + length, EventSort());

        DEVICE_SYNC_CHECK();

        start += length;
    }

    cudaDeviceSynchronize();
    timer.Stop();
    __ct_printf("%f", timer.GetMillis());

    for(CTuint i = 0; i < min(m_pool.GetStreamCount(), nodeCount); ++i)
    {
        nutty::cuStream& stream = m_pool.GetStream(i);
        nutty::cuEvent e = stream.RecordEvent();
        cudaStreamWaitEvent(0, e.GetPointer(), 0);
    }

    DEVICE_SYNC_CHECK();

    reorderEvents<<<eventGrid, eventBlock>>>(eventsDst, eventsSrc, eventCount);

    m_scannedEventTypeStartMask.Resize(m_events_Type[1].Size());
    m_eventTypeMaskSums.Resize(m_events_Type[1].Size());
    m_scannedEventTypeEndMask.Resize(m_events_Type[1].Size());

    EventStartScanOp<CTbyte> op0;
    nutty::ExclusiveScan(m_events_Type[1].Begin(), m_events_Type[1].Begin() + eventCount, m_scannedEventTypeStartMask.Begin(), m_eventTypeMaskSums.Begin(), op0);

    nutty::ZeroMem(m_eventTypeMaskSums);
    nutty::ZeroMem(m_scannedEventTypeEndMask);

    EventEndScanOp<CTbyte> op1;
    nutty::ExclusiveScan(m_events_Type[1].Begin(), m_events_Type[1].Begin() + eventCount, m_scannedEventTypeEndMask.Begin(), m_eventTypeMaskSums.Begin(), op1);
        
    DEVICE_SYNC_CHECK();
    static CTuint bla = 0;
    computeSAHSplits<<<eventGrid, eventBlock>>>(
        eventsDst,
        nodesContentCount,
        m_splits,
        m_nodesBBox[0].Begin()(),
        m_nodesContent, 
        m_scannedEventTypeStartMask.Begin()(), 
        m_scannedEventTypeEndMask.Begin()(),
        eventCount,
        bla);
    bla++;
    
#if 0
    for(int i = 0; i < eventCount && bla == 8; ++i)
    {
        ct_printf("%d [%d %d] id=%d Split=%.4f SAH=%.4f %d\n", i, m_splits_Below[i], m_splits_Above[i], 
            m_events_IndexedEdge[i].index, m_events_IndexedEdge[i].v, m_splits_IndexedSplit[i].sah, m_scannedEventTypeEndMask[i]);
    }
#endif

    start = 0;
    m_pool.Reset();
    cudaDeviceSynchronize();
    timer.Start();
    for(int i = 0; i < nodeCount; ++i)
    {
        CTuint cc = hNodesContentCount[i];
        CTuint length = 2 * cc;
#ifdef _DEBUG
        if(cc <= MAX_ELEMENTS_PER_LEAF)
        {
            assert(0 && "cc <= MAX_ELEMENTS_PER_LEAF");
            //start += length;
            continue;
        }
#endif
        IndexedSAHSplit neutralSplit;
        neutralSplit.index = 0;
        neutralSplit.sah = FLT_MAX;
        
        nutty::cuStream& stream = m_pool.PeekNextStream();
        nutty::SetStream(stream);

        nutty::Reduce(m_splits_IndexedSplit.Begin() + start, m_splits_IndexedSplit.Begin() + start + length, ReduceIndexedSplit(), neutralSplit);

        DEVICE_SYNC_CHECK();
#ifdef PRINT_OUT
        IndexedSAHSplit s = *(m_splits_IndexedSplit.Begin() + start);
        std::stringstream ss;
        ss << m_nodesBBox[0][i];
        ct_printf("%s ", ss.str().c_str());
        ct_printf("id=%d, memoryadd=%d ", s.index, start);
        CTreal plane = m_splits_Plane[s.index];
        CTbyte axis = m_splits_Axis[s.index];
        CTuint below = m_splits_Below[s.index];
        CTuint above = m_splits_Above[s.index];
        ct_printf("SPLIT= %d %f %f %d-%d\n", (CTuint)axis, plane, s.sah, below, above);

//         for(int i = start; i < start + length && IS_INVALD_SAH(s.sah); ++i)
//         {
//             ct_printf("%d [%d %d] id=%d Split=%.4f SAH=%.4f %d\n", i, m_splits_Below[i], m_splits_Above[i], 
//                 m_events_IndexedEdge[i].index, m_events_IndexedEdge[i].v, m_splits_IndexedSplit[i].sah, m_scannedEventTypeEndMask[i]);
//         }
//         if(IS_INVALD_SAH(s.sah))
//         {
//             __debugbreak();
//         }
#endif
        start += length;
    }
    cudaDeviceSynchronize();
    timer.Stop();
    __ct_printf("|%f ", timer.GetMillis());
    for(CTuint i = 0; i < min(m_pool.GetStreamCount(), nodeCount); ++i)
    {
        nutty::cuStream& stream = m_pool.GetStream(i);
        nutty::cuEvent e = stream.RecordEvent();
        cudaStreamWaitEvent(0, e.GetPointer(), 0);
    }
    
    nutty::SetDefaultStream();
}

CTuint cuKDTreeBitonicSearch::CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint nodeRange)
{
    m_leafCountScanner.Resize(nodeRange);
    m_leafCountScanner.ExcScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, TypeOp<CTbyte>());

    CTuint leafCount = m_leafCountScanner.GetPrefixSum()[nodeRange-1] + (*(isLeafBegin + nodeOffset + nodeRange - 1) == 1);

    if(!leafCount)
    {
        return 0;
    }

    //not needed m_leafCountScanner - id should do it, anyway not the problem atm
    m_interiorCountScanner.Resize(nodeRange);
    m_interiorCountScanner.ExcScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, InvTypeOp<CTbyte>());

    m_maskedInteriorContent.Resize(nodeRange);
    m_maskedleafContent.Resize(nodeRange);
    m_interiorContentScanner.Resize(nodeRange);
    m_leafContentScanner.Resize(nodeRange);
           
    CTuint block = nutty::cuda::GetCudaBlock(nodeRange, 256U);
    CTuint grid = nutty::cuda::GetCudaGrid(nodeRange, block);

    createInteriorAndLeafContentCountMasks<<<grid, block>>>(
        isLeafBegin() + nodeOffset,
        m_nodes_ContentCount.Begin()(), 
        m_maskedleafContent.Begin()(), 
        m_maskedInteriorContent.Begin()(), nodeRange);

    m_interiorContentScanner.ExcScan(m_maskedInteriorContent.Begin(), m_maskedInteriorContent.Begin() + nodeRange, nutty::PrefixSumOp<CTuint>());
    m_leafContentScanner.ExcScan(m_maskedleafContent.Begin(), m_maskedleafContent.Begin() + nodeRange, nutty::PrefixSumOp<CTuint>());

    return leafCount;
}

MakeLeavesResult cuKDTreeBitonicSearch::MakeLeaves(
    nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, 
    /*nutty::DeviceBuffer<CTbyte>::iterator& isNodeLeafFinalBegin, */
    CTuint g_nodeOffset, 
    CTuint nodeOffset, 
    CTuint nodeCount, 
    CTuint primitiveCount, 
    CTuint currentLeafCount, 
    CTuint leafContentOffset,
    CTuint initNodeToLeafIndex)
{
    CTuint leafCount = CheckRangeForLeavesAndPrepareBuffer(isLeafBegin, nodeOffset, nodeCount);

    if(!leafCount)
    {
        MakeLeavesResult result;
        result.leafCount = 0;
        result.interiorPrimitiveCount = primitiveCount;
        result.leafPrimitiveCount = 0;
        return result;
    }

    m_leafNodesContentStart.Resize(currentLeafCount + leafCount);
    m_leafNodesContentCount.Resize(currentLeafCount + leafCount);

    CTuint eventCount = 2 * primitiveCount;
    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
    CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

    m_primIsLeafScanner.Resize(primitiveCount);
    m_primIsNoLeafScanner.Resize(primitiveCount);

    m_primIsLeafScanner.ExcScan(m_primIsLeaf.Begin(), m_primIsLeaf.Begin() + primitiveCount, TypeOp<CTbyte>());
    m_primIsNoLeafScanner.ExcScan(m_primIsLeaf.Begin(), m_primIsLeaf.Begin() + primitiveCount, InvTypeOp<CTbyte>());

    if(initNodeToLeafIndex)
    {
        setNodeToLeafIndex<1><<<nodeGrid, nodeBlock>>>(
            m_nodes_NodeIdToLeafIndex.Begin()(),
            m_activeNodes.Begin()(),
            m_leafCountScanner.GetPrefixSum().Begin()(),
            m_activeNodesIsLeaf.Begin()() + nodeOffset,
            g_nodeOffset,
            currentLeafCount,
            nodeCount);
    }
    else
    {
        setNodeToLeafIndex<0><<<nodeGrid, nodeBlock>>>(
            m_nodes_NodeIdToLeafIndex.Begin()(),
            m_activeNodes.Begin()(),
            m_leafCountScanner.GetPrefixSum().Begin()(),
            m_activeNodesIsLeaf.Begin()(),
            g_nodeOffset,
            currentLeafCount,
            nodeCount);
    }
    
    compactLeafNInteriorData<<<nodeGrid, nodeBlock>>>(
        m_interiorContentScanner.GetPrefixSum().Begin()(), 
        m_leafContentScanner.GetPrefixSum().Begin()(),
        m_nodes_ContentCount.Begin()(),
        m_nodes_ContentStartAdd.Begin()(),
        m_nodesBBox[1].Begin()(),
        isLeafBegin() + nodeOffset,

        m_primIsLeafScanner.GetPrefixSum().Begin()(),
        m_leafCountScanner.GetPrefixSum().Begin()(), 
        m_interiorCountScanner.GetPrefixSum().Begin()(),
        m_activeNodes.Begin()(),
        m_newNodesContentCount.Begin()(),
        m_newNodesContentStartAdd.Begin()(),
        m_leafNodesContentStart.Begin()(),
        m_leafNodesContentCount.Begin()(),
        m_newActiveNodes.Begin()(),
        m_nodesBBox[0].Begin()(),

        currentLeafCount, leafContentOffset, nodeCount
        );

    CTuint copyDistance = nodeCount - leafCount;
    if(copyDistance)
    {
        nutty::Copy(m_nodes_ContentCount.Begin(), m_newNodesContentCount.Begin(), m_newNodesContentCount.Begin() + nodeCount);
        nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin() + nodeCount);
        nutty::Copy(m_activeNodes.Begin(), m_newActiveNodes.Begin(), m_newActiveNodes.Begin() + nodeCount);
    }

    if(currentLeafCount)
    {
        leafContentOffset = m_leafNodesContentCount[currentLeafCount-1];
        leafContentOffset += m_leafNodesContentStart[currentLeafCount-1];
    }

    CTbyte last = (*(m_primIsLeaf.Begin() + primitiveCount - 1)) ^ 1;

    CTuint interiorPrimCount = m_primIsNoLeafScanner.GetPrefixSum()[primitiveCount - 1] + last;
    CTuint leafPrimCount = primitiveCount - interiorPrimCount;
    interiorPrimCount = interiorPrimCount > primitiveCount ? 0 : interiorPrimCount;

    m_leafNodesContent.Resize(leafContentOffset + leafPrimCount);

    m_newInteriorContent.Resize(interiorPrimCount);
    m_newPrimNodeIndex.Resize(interiorPrimCount);
    m_newPrimPrefixSum.Resize(interiorPrimCount);

    CTuint block = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    CTuint grid = nutty::cuda::GetCudaGrid(primitiveCount, block);

    compactLeafNInteriorContent<<<grid, block>>>(
        m_leafNodesContent.Begin()() + leafContentOffset,
        m_newInteriorContent.Begin()(),
        m_primIsLeaf.Begin()(), 
        m_primIsLeafScanner.GetPrefixSum().Begin()(),
        m_primIsNoLeafScanner.GetPrefixSum().Begin()(),
        m_interiorCountScanner.GetPrefixSum().Begin()(),

        m_primIndex.Begin()(), 
        m_primNodeIndex.Begin()(),
        m_interiorContentScanner.GetPrefixSum().Begin()(),

        m_newPrimNodeIndex.Begin()(),
        m_newPrimPrefixSum.Begin()(),

        primitiveCount
        );
    
    nutty::Copy(m_primIndex.Begin(), m_newInteriorContent.Begin(), m_newInteriorContent.Begin() + interiorPrimCount);
    nutty::Copy(m_primNodeIndex.Begin(), m_newPrimNodeIndex.Begin(), m_newPrimNodeIndex.Begin() + interiorPrimCount);
    nutty::Copy(m_primPrefixSum.Begin(), m_newPrimPrefixSum.Begin(), m_newPrimPrefixSum.Begin() + interiorPrimCount);

    MakeLeavesResult result;
    result.leafCount = leafCount;
    result.interiorPrimitiveCount = interiorPrimCount;
    result.leafPrimitiveCount = leafPrimCount;

    return result;
}

CT_RESULT cuKDTreeBitonicSearch::Update(void)
{
    if(!m_initialized)
    {
        InitBuffer();
        m_initialized = true;
    }

    ClearBuffer();
    
    CTuint primitiveCount = m_currentTransformedVertices.Size() / 3;

    m_nodes_ContentCount.Insert(0, primitiveCount);

    uint _block = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    uint _grid = nutty::cuda::GetCudaGrid(primitiveCount, _block);

    cudaCreateTriangleAABBs(m_currentTransformedVertices.Begin()(), m_primAABBs.Begin()(), primitiveCount);
   
    DEVICE_SYNC_CHECK();

    initNodesContent<<<_grid, _block>>>(m_nodesContent, primitiveCount);

    DEVICE_SYNC_CHECK();

    static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
    static float3 min3f = -max3f;

    BBox bboxN;
    bboxN.m_min = max3f; 
    bboxN.m_max = min3f;
    m_sceneBBox.Resize(m_primAABBs.Size()/2);
    nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), ReduceBBox(), bboxN);

    DEVICE_SYNC_CHECK(); 

    nutty::Copy(m_nodesBBox[0].Begin(), m_sceneBBox.Begin(), 1);

    m_nodes_IsLeaf.Insert(0, m_depth == 0);

    CTuint g_interiorNodesCountOnThisLevel = 1;
    CTuint g_currentInteriorNodesCount = 1;
    CTuint g_currentLeafCount = 0;
    CTuint g_leafContentOffset = 0;
    CTuint g_childNodeOffset = 1;
    CTuint g_childNodeOffset2 = 1;
    CTuint g_nodeOffset = 0;
    CTuint g_nodeOffset2 = 0;
    CTuint g_lastChildCount = 0;
    CTuint g_entries = 1;
    
    m_activeNodes.Insert(0, 0);
    m_nodes_NodeIdToLeafIndex.Insert(0, 0);
    nutty::ZeroMem(m_nodes_NodeIdToLeafIndex);
    double time = 0;
    for(CTbyte d = 0; d <= m_depth; ++d)
    {
        ct_printf("\nNew Level=%d (%d)\n", d, m_depth);
        nutty::ZeroMem(m_activeNodesIsLeaf);
        CTuint nodeCount = g_interiorNodesCountOnThisLevel;
        CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
        CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        CTuint eventCount = 2 * primitiveCount;
        CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
        CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

        DEVICE_SYNC_CHECK();

        m_hNodesContentCount.Resize(nodeCount);
        nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);
        
        PRINT_BUFFER_N(m_hNodesContentCount, nodeCount);
//         PRINT_BUFFER_N(m_nodes_ContentStartAdd, nodeCount);
//         PRINT_BUFFER(m_activeNodes);

        m_pool.ClearEvents();
        chimera::util::HTimer timer;
        cudaDeviceSynchronize();
        timer.Start();
        ComputeSAH_Splits(
            nodeCount, 
            primitiveCount, 
            m_hNodesContentCount.Begin()(),
            m_nodes_ContentCount.Begin()(),
            m_nodes_IsLeaf.Begin()() + g_nodeOffset, 
            m_nodesContent);
        cudaDeviceSynchronize();
        timer.Stop();
        time += timer.GetMillis();
        __ct_printf("%f (%f) primitiveCount=%d nodeCount=%d\n", time, timer.GetMillis(), primitiveCount, nodeCount);
        makeLeafIfBadSplitOrLessThanMaxElements<<<nodeGrid, nodeBlock>>>(
            m_nodes,
            m_nodes_IsLeaf.Begin()() + g_nodeOffset,
            m_activeNodes.Begin()(),
            m_activeNodesIsLeaf.Begin()(), 
            m_splits,
            d == m_depth-1,
            nodeCount);

        CTuint b = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
        CTuint g = nutty::cuda::GetCudaGrid(primitiveCount, b);
        nutty::ZeroMem(m_primIsLeaf);

        setPrimBelongsToLeaf<<<g, b>>>(m_nodesContent, m_activeNodesIsLeaf.Begin()(), primitiveCount);

        m_newNodesContentCount.Resize(m_nodes_ContentCount.Size());
        m_newNodesContentStartAdd.Resize(m_nodes_ContentCount.Size());

        m_lastNodeContentStartAdd.Resize(m_newNodesContentStartAdd.Size());
        nutty::Copy(m_lastNodeContentStartAdd.Begin(), m_nodes_ContentStartAdd.Begin(), m_nodes_ContentStartAdd.Begin() + nodeCount);

        MakeLeavesResult leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_nodeOffset, 0, nodeCount, primitiveCount, g_currentLeafCount, g_leafContentOffset, 0);

        CTuint lastLeaves = leavesRes.leafCount;
        CTuint lastCnt = primitiveCount;
        primitiveCount = leavesRes.interiorPrimitiveCount;

        if(leavesRes.interiorPrimitiveCount)
        {
            g_leafContentOffset += leavesRes.leafPrimitiveCount;

            classifyEdges<<<eventGrid, eventBlock>>>(m_nodes, m_events[1], m_splits, m_activeNodesIsLeaf.Begin()(), m_edgeMask.Begin()(), m_lastNodeContentStartAdd.Begin()(), eventCount);
            DEVICE_SYNC_CHECK();

            nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.Begin() + eventCount, m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());
            DEVICE_SYNC_CHECK();

            primitiveCount = m_scannedEdgeMask[eventCount - 1] + m_edgeMask[eventCount - 1];

            nutty::ZeroMem(m_primIsLeaf);
            nutty::DeviceBuffer<CTuint> blabla(primitiveCount);
            setPrimBelongsToLeafFromEvents<<<eventGrid, eventBlock>>>(
                m_events[1], 
                m_nodesContent,
                m_activeNodesIsLeaf.Begin()() + nodeCount, 
                m_edgeMask.Begin()(),
                m_scannedEdgeMask.Begin()(), 
                eventCount,
                blabla.Begin()());

            if(lastLeaves)
            {
                setActiveNodesMask<1><<<nodeGrid, nodeBlock>>>(
                    m_activeNodesThisLevel.Begin()(), 
                    m_activeNodesIsLeaf.Begin()(), 
                    m_interiorCountScanner.GetPrefixSum().Begin()(),
                    0, 
                    nodeCount);
            }
            else
            {
               setActiveNodesMask<0><<<nodeGrid, nodeBlock>>>(
                    m_activeNodesThisLevel.Begin()(), 
                    m_activeNodesIsLeaf.Begin()(), 
                    m_interiorCountScanner.GetPrefixSum().Begin()(),
                    0, 
                    nodeCount);
            }

            CTuint childCount = (nodeCount - leavesRes.leafCount) * 2;
            CTuint thisLevelNodesLeft = nodeCount - leavesRes.leafCount;

            nodeBlock = nutty::cuda::GetCudaBlock(thisLevelNodesLeft, 256U);
            nodeGrid = nutty::cuda::GetCudaGrid(thisLevelNodesLeft, nodeBlock);

            initThisLevelInteriorNodes<<<nodeGrid, nodeBlock>>>(
                m_nodes,
                m_splits,

                m_leafNodesContentCount.Begin()(),
                m_leafNodesContentStart.Begin()(),

                m_scannedEdgeMask.Begin()(),
                m_interiorCountScanner.GetPrefixSum().Begin()(),
                m_primIsLeafScanner.GetPrefixSum().Begin()(),
                m_activeNodes.Begin()(),
                m_activeNodesIsLeaf.Begin()(),
                m_activeNodesThisLevel.Begin()(),
                m_nodesBBox[0].Begin()(), 
                m_nodesBBox[1].Begin()(), 

                m_newNodesContentCount.Begin()(),
                m_newNodesContentStartAdd.Begin()(),
                m_newActiveNodes.Begin()(),
                m_activeNodesIsLeaf.Begin()() + nodeCount,
                g_childNodeOffset,
                g_childNodeOffset2,
                g_nodeOffset,
                leavesRes.leafCount,
                thisLevelNodesLeft,
                m_lastNodeContentStartAdd.Begin()(),
                m_depth == d+1);

            nutty::Copy(m_activeNodes.Begin(), m_newActiveNodes.Begin(), m_newActiveNodes.Begin() + childCount);
            nutty::Copy(m_nodes_ContentCount.Begin(), m_newNodesContentCount.Begin(), m_newNodesContentCount.Begin() + childCount);
            nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin() + childCount);

            compactPrimitivesFromEvents<<<eventGrid, eventBlock>>>(
                m_events[1], 
                m_nodes, 
                m_nodesContent,
                m_leafCountScanner.GetPrefixSum().Begin()(),
                m_edgeMask.Begin()(), 
                m_scannedEdgeMask.Begin()(),
                d, eventCount, lastLeaves);

            nodeBlock = nutty::cuda::GetCudaBlock(2 * nodeCount, 256U);
            nodeGrid = nutty::cuda::GetCudaGrid(2 * nodeCount, nodeBlock);
            setNodeToLeafIndex<2><<<nodeGrid, nodeBlock>>>(
                m_nodes_NodeIdToLeafIndex.Begin()(),
                m_activeNodes.Begin()(),
                m_leafCountScanner.GetPrefixSum().Begin()(),
                m_activeNodesIsLeaf.Begin()(),
                g_childNodeOffset,
                g_currentLeafCount,
                2 * nodeCount);

            leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_childNodeOffset, nodeCount, childCount, primitiveCount, g_currentLeafCount + lastLeaves, g_leafContentOffset, 1);

            primitiveCount = leavesRes.interiorPrimitiveCount;

        }
        else
        {
            //todo
            for(int i = 0; i < nodeCount; ++i)
            {
                m_nodes_IsLeaf.Insert(g_nodeOffset + i, (CTbyte)1);
            }
        }
        g_entries += 2 * nodeCount;
        g_lastChildCount = 2 * nodeCount;
        g_nodeOffset2 = g_nodeOffset;
        g_interiorNodesCountOnThisLevel = 2 * (nodeCount - lastLeaves) - leavesRes.leafCount;
        g_currentInteriorNodesCount += g_interiorNodesCountOnThisLevel;
        g_nodeOffset = g_childNodeOffset;
        g_childNodeOffset += 2 * (nodeCount);

        //ct_printf("%d %d\n", g_nodeOffset, g_childNodeOffset);
        //update globals
        g_leafContentOffset += leavesRes.leafPrimitiveCount;
        
        g_currentLeafCount += lastLeaves + leavesRes.leafCount;

        ct_printf(
            "g_nodeOffset=%d g_childNodeOffset=%d g_leafContentOffset=%d g_interiorNodesCountOnThisLevel=%d g_currentInteriorNodesCount=%d g_currentLeafCount=%d\nCreated '%d' Leaves, Interior Nodes '%d'\n", 
              g_nodeOffset, g_childNodeOffset, g_leafContentOffset, g_interiorNodesCountOnThisLevel, g_currentInteriorNodesCount, g_currentLeafCount, lastLeaves + leavesRes.leafCount, g_interiorNodesCountOnThisLevel);

        DEVICE_SYNC_CHECK();

        if(!leavesRes.leafCount)
        {
            nutty::Copy(m_nodesBBox[0].Begin(), m_nodesBBox[1].Begin(), m_nodesBBox[1].Begin() + g_interiorNodesCountOnThisLevel);
        }
        PRINT_BUFFER(m_nodesBBox[0]);
        
//         PRINT_BUFFER_N(m_primNodeIndex, primitiveCount);
//         PRINT_BUFFER_N(m_nodes_LeftChild, g_entries);
//         PRINT_BUFFER_N(m_nodes_RightChild, g_entries);
//         PRINT_BUFFER_N(m_nodes_IsLeaf, g_entries);
//         PRINT_BUFFER_N(m_nodes_NodeIdToLeafIndex, g_entries);

        if(primitiveCount == 0 || g_interiorNodesCountOnThisLevel == 0) //all nodes are leaf nodes
        {
            primitiveCount = lastCnt;
            break;
        }
        
        if(d < m_depth-1) //are we not done?
        {
            //check if we need more memory
            if(2 * primitiveCount > m_edgeMask.Size())
            {
                GrowPrimitiveEventMemory();
            }

            if(m_activeNodes.Size() < g_interiorNodesCountOnThisLevel + 2 * g_interiorNodesCountOnThisLevel)
            {
                GrowPerLevelNodeMemory(4 * 2 * g_interiorNodesCountOnThisLevel);
            }

            if(m_nodes_IsLeaf.Size() < (g_childNodeOffset + 2 * g_interiorNodesCountOnThisLevel))
            {
                GrowNodeMemory();
            }
        }
    }

    m_interiorNodesCount = g_currentInteriorNodesCount;
    m_leafNodesCount = g_currentLeafCount;
    CTuint allNodeCount = m_interiorNodesCount + m_leafNodesCount;

#ifdef _DEBUG
    ValidateTree();
#endif
     //m_nodes_NodeIdToLeafIndex.Resize(allNodeCount);
    // nutty::ExclusivePrefixSumScan(m_nodes_IsLeaf.Begin(), m_nodes_IsLeaf.Begin() + allNodeCount, m_nodes_NodeIdToLeafIndex.Begin(), m_edgeMaskSums.Begin());

    ct_printf("Tree Summary:\n");
    PRINT_BUFFER(m_nodes_IsLeaf);
    PRINT_BUFFER(m_nodes_Split);
    PRINT_BUFFER(m_nodes_SplitAxis);
    PRINT_BUFFER(m_nodes_LeftChild);
    PRINT_BUFFER(m_nodes_RightChild);
    PRINT_BUFFER(m_leafNodesContentCount);
    PRINT_BUFFER(m_leafNodesContentStart);
    PRINT_BUFFER(m_nodes_NodeIdToLeafIndex);

    if(m_leafNodesContent.Size() < 1024)
    {
        PRINT_BUFFER(m_leafNodesContent);
    }
    else
    {
        ct_printf("skipping content '%d' elements...\n", m_leafNodesContent.Size());
    }

    DEVICE_SYNC_CHECK();

    return CT_SUCCESS;
}

void cuKDTreeBitonicSearch::ValidateTree(void)
{
    std::queue<CTuint> queue;

    queue.push(0);

    while(!queue.empty())
    {
        CTuint node = queue.front();
        queue.pop();
        ct_printf("%d ", node);
        if(!m_nodes_IsLeaf[node])
        {
            ct_printf("\n");
             //assertions are happening here if we are out of bounds
            CTuint left = m_nodes_LeftChild[node];
            CTuint right = m_nodes_RightChild[node];
            if(left < node || right < node)
            {
                assert(0 && "fuck");
            }
            queue.push(left);
            queue.push(right);
        }
        else
        {
            CTuint leafIndex = m_nodes_NodeIdToLeafIndex[node];
            ct_printf(" - %d\n", leafIndex);
        }
    }
}