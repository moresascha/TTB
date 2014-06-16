
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

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
        bbox._min = fminf(t0._min, t1._min);
        bbox._max = fmaxf(t0._max, t1._max);
        return bbox;
    }
};

void cuKDTreeBitonicSearch::InitBuffer(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;

    m_depth = (byte)min(32, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));

//     CTuint maxlevelInteriorNodesCount = elemsBeforeLevel(m_depth);
// 
//     CTuint maxLeafNodesCount = elemsOnLevel(m_depth);
// 
//     CTuint maxNodeCount = maxlevelInteriorNodesCount + maxLeafNodesCount;

    m_primAABBs.Resize(primitiveCount); nutty::ZeroMem(m_primAABBs);

    //m_sceneBBox.Resize(1); nutty::ZeroMem(m_sceneBBox);

    GrowNodeMemory();
    GrowPerLevelNodeMemory();
    GrowMemory();

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

void cuKDTreeBitonicSearch::GrowPerLevelNodeMemory(void)
{
    size_t newSize = m_nodes_IsLeaf.Size() ? m_nodes_IsLeaf.Size() * 2 : 32;

    m_activeNodesIsLeaf.Resize(newSize);
    m_activeNodes.Resize(newSize);
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
    size_t newSize = m_nodes_IsLeaf.Size() ? m_nodes_IsLeaf.Size() * 2 : 32;

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

void cuKDTreeBitonicSearch::GrowMemory(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;
    uint eventCount = 4 * primitiveCount; //4 times as big
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

template <
    typename T
>
struct InvTypeOp
{
    __device__ T operator()(T elem)
    {
        return elem ^ 1;
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
        return elem;
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
        return 0;
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

void cuKDTreeBitonicSearch::PrintStatus(void)
{
    PRINT_BUFFER(m_primNodeIndex);
    PRINT_BUFFER(m_primIndex);
    PRINT_BUFFER(m_primPrefixSum);
    PRINT_BUFFER(m_nodes_ContentCount);
    PRINT_BUFFER(m_nodes_ContentStartAdd);
    PRINT_BUFFER(m_nodes_Split);
    PRINT_BUFFER(m_nodes_SplitAxis);
    PRINT_BUFFER(m_nodes_IsLeaf);
    PRINT_BUFFER(m_leafNodesContent);
    PRINT_BUFFER(m_leafNodesContentCount);
    PRINT_BUFFER(m_leafNodesContentStart);
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

    nutty::ZeroMem(m_events_NodeIndex[0]);

    CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
    CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

    CTuint eventCount = 2 * primitiveCount;
    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    Event eventsSrc = m_events[0];
    Event eventsDst = m_events[1];

    createEvents<<<elementGrid, elementBlock>>>(eventsSrc, m_primAABBs.Begin()(), m_nodesBBox[0].Begin()(), nodesContent, primitiveCount);

    DEVICE_SYNC_CHECK();

    CTuint start = 0;

    for(int i = 0; i < nodeCount; ++i)
    {
        CTuint cc = hNodesContentCount[i];
        CTuint length = 2 * cc;

        if(cc <= MAX_ELEMENTS_PER_LEAF)
        {
            assert(0 && "cc <= MAX_ELEMENTS_PER_LEAF");
            //start += length;
            continue;
        }

        nutty::Sort(m_events_IndexedEdge.Begin() + start, m_events_IndexedEdge.Begin() + start + length, EventSort());

        DEVICE_SYNC_CHECK();

        start += length;
    }

    DEVICE_SYNC_CHECK();

    reorderEvents<<<eventGrid, eventBlock>>>(eventsDst, eventsSrc, eventCount);

    m_scannedEventTypeStartMask.Resize(m_events_Type[1].Size());
    m_eventTypeMaskSums.Resize(m_events_Type[1].Size());
    m_scannedEventTypeEndMask.Resize(m_events_Type[1].Size());

    EventStartScanOp<CTbyte> op0;
    nutty::InclusiveScan(m_events_Type[1].Begin(), m_events_Type[1].Begin() + eventCount, m_scannedEventTypeStartMask.Begin(), m_eventTypeMaskSums.Begin(), op0);

    nutty::ZeroMem(m_eventTypeMaskSums);
    nutty::ZeroMem(m_scannedEventTypeEndMask);

    EventEndScanOp<CTbyte> op1;
    nutty::InclusiveScan(m_events_Type[1].Begin(), m_events_Type[1].Begin() + eventCount, m_scannedEventTypeEndMask.Begin(), m_eventTypeMaskSums.Begin(), op1);

//     PRINT_BUFFER(m_scannedEventTypeStartMask);
//     PRINT_BUFFER(m_scannedEventTypeEndMask);
    DEVICE_SYNC_CHECK();

    computeSAHSplits<<<eventGrid, eventBlock>>>(
        eventsDst,
        nodesContentCount,
        m_splits,
        m_nodesBBox[0].Begin()(),
        m_nodesContent, 
        m_scannedEventTypeStartMask.Begin()(), 
        m_scannedEventTypeEndMask.Begin()(),
        eventCount);

    start = 0;
#if 1
    for(int i = 0; i < eventCount; ++i)
    {
        ct_printf("%d [%d %d] id=%d Split=%.4f SAH=%.4f\n", i, m_splits_Below[i], m_splits_Above[i], 
            m_events_IndexedEdge[i].index, m_events_IndexedEdge[i].v, m_splits_IndexedSplit[i].sah);
    }
#endif

    for(int i = 0; i < nodeCount; ++i)
    {
        CTuint cc = hNodesContentCount[i];
        CTuint length = 2 * cc;

        if(cc <= MAX_ELEMENTS_PER_LEAF)
        {
            assert(0 && "cc <= MAX_ELEMENTS_PER_LEAF");
            //start += length;
            continue;
        }

        IndexedSAHSplit neutralSplit;
        neutralSplit.index = 0;
        neutralSplit.sah = FLT_MAX;
            
        nutty::Reduce(m_splits_IndexedSplit.Begin() + start, m_splits_IndexedSplit.Begin() + start + length, ReduceIndexedSplit(), neutralSplit);
        DEVICE_SYNC_CHECK();

        IndexedSAHSplit s = *(m_splits_IndexedSplit.Begin() + start);
        ct_printf("id=%d, ", s.index);
        CTreal plane = m_splits_Plane[s.index];
        CTbyte axis = m_splits_Axis[s.index];
        CTuint below = m_splits_Below[s.index];
        CTuint above = m_splits_Above[s.index];
        ct_printf("SPLIT= %d %f %f %d-%d\n", (CTuint)axis, plane, s.sah, below, above);

        start += length;
    }
}

CTuint cuKDTreeBitonicSearch::CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint nodeRange)
{
    m_leafCountScanner.Resize(nodeRange);
    m_leafCountScanner.ExcScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, TypeOp<CTbyte>());
    CTuint leafCount = m_leafCountScanner.GetPrefixSum()[nodeRange-1] + *(isLeafBegin + nodeOffset + nodeRange - 1);

    if(!leafCount)
    {
        return 0;
    }

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

MakeLeavesResult cuKDTreeBitonicSearch::MakeLeaves(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint nodeCount, CTuint primitiveCount, CTuint currentLeafCount, CTuint leafContentOffset)
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
        m_newNodesContentCount.Begin()(),
        m_newNodesContentStartAdd.Begin()(),
        m_leafNodesContentStart.Begin()(),
        m_leafNodesContentCount.Begin()(),
        m_activeNodes.Begin()(),
        m_nodesBBox[0].Begin()(),

        currentLeafCount, leafContentOffset, nodeCount
        );

    nutty::Copy(m_nodes_ContentCount.Begin(), m_newNodesContentCount.Begin(), m_newNodesContentCount.Begin() + 2 * nodeCount);
    nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin() + 2 * nodeCount);

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
    PRINT_BUFFER(m_primNodeIndex);
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
    
    PRINT_BUFFER(m_newPrimPrefixSum);
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
    bboxN._min = max3f; 
    bboxN._max = min3f;
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
    CTuint g_nodeOffset = 0;

    m_activeNodes.Insert(0, 0);

    for(CTbyte d = 0; d < m_depth; ++d)
    {
        ct_printf("\nNew Level\n");
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

        PRINT_BUFFER(m_hNodesContentCount);

        ComputeSAH_Splits(
            nodeCount, 
            primitiveCount, 
            m_hNodesContentCount.Begin()(),
            m_nodes_ContentCount.Begin()(),
            m_nodes_IsLeaf.Begin()() + g_nodeOffset, 
            m_nodesContent);

        PRINT_BUFFER(m_nodes_IsLeaf);

        makeLealIfBadSplitOrLessThanMaxElements<<<nodeGrid, nodeCount>>>(
            m_nodes,
            m_activeNodesIsLeaf.Begin()(), 
            m_splits, 
            nodeCount);

        CTuint b = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
        CTuint g = nutty::cuda::GetCudaGrid(primitiveCount, eventBlock);

        setPrimBelongsToLeaf<<<g, b>>>(m_nodesContent, m_activeNodes.Begin()(), m_activeNodesIsLeaf.Begin()(), primitiveCount);

        classifyEdges<<<eventGrid, eventBlock>>>(m_nodes, m_events[1], m_splits, m_activeNodesIsLeaf.Begin()(), m_edgeMask.Begin()(), eventCount);
        DEVICE_SYNC_CHECK();

        nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.Begin() + eventCount, m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());
        DEVICE_SYNC_CHECK();

        CTuint lastCnt = primitiveCount;
        primitiveCount = m_scannedEdgeMask[eventCount - 1] + m_edgeMask[eventCount - 1];

        MakeLeavesResult leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), 0, nodeCount, primitiveCount, g_currentLeafCount, g_leafContentOffset);
        primitiveCount = leavesRes.interiorPrimitiveCount;
        g_leafContentOffset += leavesRes.leafPrimitiveCount;

        nutty::ZeroMem(m_primIsLeaf);

        setPrimBelongsToLeafFromEvents<<<eventGrid, eventBlock>>>(
            m_events[1], 
            m_nodesContent,
            m_activeNodesIsLeaf.Begin()() + nodeCount, 
            m_edgeMask.Begin()(),
            m_scannedEdgeMask.Begin()(), 
            eventCount);
        PRINT_BUFFER(m_activeNodesIsLeaf);

        CTuint childCount = 2 * nodeCount;
    
        m_newNodesContentCount.Resize(m_nodes_ContentCount.Size());
        m_newNodesContentStartAdd.Resize(m_nodes_ContentCount.Size());

        initThisLevelInteriorNodes<<<nodeGrid, nodeBlock>>>(
            m_nodes,
            m_splits,

            m_leafNodesContentCount.Begin()(),
            m_leafNodesContentStart.Begin()(),

            m_scannedEdgeMask.Begin()(),
            m_primIsLeafScanner.GetPrefixSum().Begin()(),
            m_activeNodes.Begin()(),
            m_activeNodesIsLeaf.Begin()(),
            m_nodesBBox[0].Begin()(), 
            m_nodesBBox[1].Begin()(), 

            m_newNodesContentCount.Begin()(),
            m_newNodesContentStartAdd.Begin()(),

            g_childNodeOffset,
            g_nodeOffset,
            leavesRes.leafCount,
            nodeCount);

        nutty::Copy(m_nodes_ContentCount.Begin(), m_newNodesContentCount.Begin(), m_newNodesContentCount.Begin() + childCount);
        nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin() + childCount);

        compactPrimitivesFromEvents<<<eventGrid, eventBlock>>>(
            m_events[1], 
            m_nodes, 
            m_nodesContent,
            m_edgeMask.Begin()(), 
            m_scannedEdgeMask.Begin()(),
            d, eventCount);

        CTuint lastLeaves = leavesRes.leafCount;

        leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), nodeCount, childCount, primitiveCount, g_currentLeafCount + lastLeaves, g_leafContentOffset);

        primitiveCount = leavesRes.interiorPrimitiveCount;

        nodeBlock = nutty::cuda::GetCudaBlock(childCount, 256U);
        nodeGrid = nutty::cuda::GetCudaGrid(childCount, nodeBlock);

        if(leavesRes.leafCount)
        {
            setActiveNodesMask<1><<<nodeGrid, nodeBlock>>>(
                m_activeNodes.Begin()(), 
                m_nodes_IsLeaf.Begin()(), 
                m_interiorCountScanner.GetPrefixSum().Begin()(), 
                g_childNodeOffset, 
                childCount);
        }
        else
        {
            setActiveNodesMask<0><<<nodeGrid, nodeBlock>>>(
                m_activeNodes.Begin()(), 
                m_nodes_IsLeaf.Begin()(), 
                m_interiorCountScanner.GetPrefixSum().Begin()(), 
                g_childNodeOffset, 
                childCount);
        }
        PRINT_BUFFER(m_activeNodes);
        PrintStatus();
        g_nodeOffset = g_childNodeOffset;
        g_childNodeOffset += 2 * nodeCount;
        //update globals
        g_leafContentOffset += leavesRes.leafPrimitiveCount;
        g_interiorNodesCountOnThisLevel = 2 * nodeCount - leavesRes.leafCount;
        g_currentInteriorNodesCount += g_interiorNodesCountOnThisLevel;
        g_currentLeafCount += lastLeaves + leavesRes.leafCount;

        DEVICE_SYNC_CHECK();

        nutty::Copy(m_nodesBBox[0].Begin(), m_nodesBBox[1].Begin(), m_nodesBBox[1].Begin() + 2 * g_interiorNodesCountOnThisLevel);

        if(primitiveCount == 0) //all nodes are leaf nodes
        {
            primitiveCount = lastCnt;
            ct_printf("interiorPrimCount == 0, Bailed out...\n");
            assert(L"0" && L"Make all nodes leaves");
            break;
        }
        
        if(d < m_depth-1) //are we not done?
        {
            //check if we need more memory
            if(2 * primitiveCount > m_edgeMask.Size())
            {
                GrowMemory();
            }

            if(m_nodes_ContentStartAdd.Size() < g_interiorNodesCountOnThisLevel)
            {
                GrowPerLevelNodeMemory();
            }

            if(m_nodes_IsLeaf.Size() < (g_currentInteriorNodesCount + 2 * g_interiorNodesCountOnThisLevel))
            {
                GrowNodeMemory();
            }
        }
        else
        {
            assert(L"0" && L"Make all nodes leaves");
        }
    }

    m_nodes_NodeIdToLeafIndex.Resize(g_currentInteriorNodesCount + g_currentLeafCount);
    nutty::ExclusivePrefixSumScan(m_nodes_IsLeaf.Begin(), m_nodes_IsLeaf.Begin() + g_currentInteriorNodesCount + g_currentLeafCount, m_nodes_NodeIdToLeafIndex.Begin(), m_edgeMaskSums.Begin());

    m_interiorNodesCount = g_currentInteriorNodesCount;
    m_leafNodesCount = g_currentLeafCount;

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