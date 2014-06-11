
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

void cuKDTreeBitonicSearch::ComputeSAH_Splits(
    CTuint nodeCount,
    CTuint nodeOffset,
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
    
    nutty::Copy(m_hIsLeaf.Begin(), m_nodes_IsLeaf.Begin() + nodeOffset, nodeCount);
    PRINT_BUFFER(m_hIsLeaf);
    PRINT_BUFFER(m_hNodesContentCount);

    DEVICE_SYNC_CHECK();

    CTuint start = 0;


    for(int i = 0; i < nodeCount; ++i)
    {
        CTuint cc = hNodesContentCount[i];
        CTuint length = 2 * cc;

        if(m_hIsLeaf[i])
        {
            //assert(0 && "cc <= MAX_ELEMENTS_PER_LEAF");
            start += length;
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

    TypeOp<CTbyte> op0;
    nutty::InclusiveScan(m_events_Type[1].Begin(), m_events_Type[1].Begin() + eventCount, m_scannedEventTypeStartMask.Begin(), m_eventTypeMaskSums.Begin(), op0);

    nutty::ZeroMem(m_eventTypeMaskSums);
    nutty::ZeroMem(m_scannedEventTypeEndMask);

    InvTypeOp<CTbyte> op1;
    nutty::InclusiveScan(m_events_Type[1].Begin(), m_events_Type[1].Begin() + eventCount, m_scannedEventTypeEndMask.Begin(), m_eventTypeMaskSums.Begin(), op1);

    DEVICE_SYNC_CHECK();

    computeSAHSplits<<<eventGrid, eventBlock>>>(
        eventsDst,
        nodesContentCount,
        isLeaf,
        m_splits,
        m_nodesBBox[0].Begin()(),
        m_nodesContent, 
        m_scannedEventTypeStartMask.Begin()(), 
        m_scannedEventTypeEndMask.Begin()(), 
        eventCount);

    start = 0;
#if 0
    for(int i = 0; i < eventCount; ++i)
    {
        CTbyte t = m_events_Type[1][i] ^ 1;
        CTuint primCount = m_nodes_ContentCount[m_events_NodeIndex[1][i]];
        CTuint prefixSum = m_events_PrefixSum[1][i]/2;
        ct_printf("%d %d, [%d %d] id=%d Split=%.4f SAH=%.4f\n", i, primCount, m_splits_Below[i], m_splits_Above[i], 
            m_events_IndexedEdge[i].index, m_events_IndexedEdge[i].v, m_splits_IndexedSplit[i].sah);
    }
#endif

    for(int i = 0; i < nodeCount; ++i)
    {
        CTuint cc = hNodesContentCount[i];
        CTuint length = 2 * cc;

        if(m_hIsLeaf[i])
        {
            //assert(0 && "cc <= MAX_ELEMENTS_PER_LEAF");
            start += length;
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

CTuint cuKDTreeBitonicSearch::GetLeavesOnLevel(CTuint nodeOffset, CTuint nodeCount)
{
    CTuint potentialLeafCount = 2 * nodeCount;
    m_leafCountScanner.Resize(potentialLeafCount);
    m_leafCountScanner.ExcScan(m_nodes_IsLeaf.Begin() + nodeOffset, m_nodes_IsLeaf.Begin() + nodeOffset + potentialLeafCount, TypeOp<CTbyte>());
    return m_leafCountScanner.GetPrefixSum()[potentialLeafCount-1] + *(m_nodes_IsLeaf.Begin() + nodeOffset + potentialLeafCount - 1);
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

    CTuint levelInteriorNodesCount = 1;
    CTuint currentInteriorNodesCount = 1;
    CTuint currentLeafCount = 0;
    CTuint lastLeafCount = 0;
    CTuint leafContentOffset = 0;
    CTuint nodeOffset = 0;

    for(CTbyte d = 0; d < m_depth; ++d)
    {
//         nutty::ZeroMem(m_edgeMask);
//         nutty::ZeroMem(m_scannedEdgeMask);

        ct_printf("----\n");

        CTuint nodeCount = levelInteriorNodesCount;
        CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
        CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        CTuint eventCount = 2 * primitiveCount;
        CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
        CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

        makeLeafFromIfMaxPrims<<<nodeGrid, nodeBlock>>>(
            m_nodes_ContentCount.Begin()(), 
            m_nodes_IsLeaf.Begin()() + nodeOffset, 
            MAX_ELEMENTS_PER_LEAF, 
            nodeCount);

        CTuint leafCount = 0;//GetLeavesOnLevel(nodeOffset, nodeCount);
        
        DEVICE_SYNC_CHECK();

        m_hNodesContentCount.Resize(nodeCount);
        m_hIsLeaf.Resize(nodeCount);
        nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);

        ComputeSAH_Splits(nodeCount, nodeOffset, primitiveCount, m_hNodesContentCount.Begin()(), m_nodes_ContentCount.Begin()(), m_nodes_IsLeaf.Begin()() + nodeOffset, m_nodesContent);

        DEVICE_SYNC_CHECK();

        classifyEdges<<<eventGrid, eventBlock>>>(m_nodes, m_events[1], m_splits, m_edgeMask.Begin()(), nodeOffset, eventCount);
        DEVICE_SYNC_CHECK();

        nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.End(), m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());
        DEVICE_SYNC_CHECK();

        if(d == m_depth-1)
        {
            initNodes<1><<<nodeGrid, nodeBlock>>>(
                m_nodes, 
                m_splits,
                m_scannedEdgeMask.Begin()(), 
                m_nodesBBox[0].Begin()(), 
                m_nodesBBox[1].Begin()(), 
                currentInteriorNodesCount,
                currentLeafCount, 
                nodeOffset,
                nodeCount);

            leafCount = levelInteriorNodesCount;
        }
        else
        {
            initNodes<0><<<nodeGrid, nodeBlock>>>(
                m_nodes,
                m_splits, 
                m_scannedEdgeMask.Begin()(), 
                m_nodesBBox[0].Begin()(), 
                m_nodesBBox[1].Begin()(), 
                currentInteriorNodesCount, 
                currentLeafCount, 
                nodeOffset,
                nodeCount);

            m_leafCountScanner.Resize(nodeCount);
            m_leafCountScanner.ExcScan(m_nodes_IsLeaf.Begin() + nodeOffset, m_nodes_IsLeaf.Begin() + nodeOffset + nodeCount, TypeOp<CTbyte>());

            leafCount = m_leafCountScanner.GetPrefixSum()[nodeCount-1] + *(m_nodes_IsLeaf.Begin() + nodeOffset + nodeCount - 1);
        }

        CTuint lastCnt = primitiveCount;
        primitiveCount = m_scannedEdgeMask[eventCount - 1] + m_edgeMask[eventCount - 1];

        if(leafCount)
        {
            nutty::ZeroMem(m_primIsLeaf);

            m_interiorCountScanner.Resize(nodeCount);
            m_interiorCountScanner.ExcScan(m_nodes_IsLeaf.Begin() + nodeOffset, m_nodes_IsLeaf.Begin() + nodeOffset + nodeCount, InvTypeOp<CTbyte>());

            setPrimBelongsToLeaf<<<eventGrid, eventBlock>>>(m_events[1], m_nodesContent, m_nodes_IsLeaf.Begin()() + nodeOffset, m_edgeMask.Begin()(), m_scannedEdgeMask.Begin()(), d, eventCount);
            m_primIsLeafScanner.Resize(primitiveCount);
            m_primIsNoLeafScanner.Resize(primitiveCount);

            TypeOp<CTbyte> ss;
            m_primIsLeafScanner.ExcScan(m_primIsLeaf.Begin(), m_primIsLeaf.Begin() + primitiveCount, ss);

            fixContentStartAddr<<<nodeGrid, nodeBlock>>>(m_nodes, m_primIsLeafScanner.GetPrefixSum().Begin()(), m_nodes_IsLeaf.Begin()() + nodeOffset, d, nodeCount);
        }

        compactPrimitivesFromEvents<<<eventGrid, eventBlock>>>(
            m_events[1], 
            m_nodes, 
            m_nodesContent,
            m_edgeMask.Begin()(), 
            m_scannedEdgeMask.Begin()(),
            d, eventCount);

        DEVICE_SYNC_CHECK();
        
        CTuint interiorPrimCount = primitiveCount;
        CTuint leafPrimCount = 0;

        if(leafCount)
        {
            InvTypeOp<CTbyte> _ss;
            m_primIsNoLeafScanner.ExcScan(m_primIsLeaf.Begin(), m_primIsLeaf.Begin() + primitiveCount, _ss);

            CTuint lastLeafCount = currentLeafCount;
            CTuint contentStartOffset = 0;

            CTbyte last = (*(m_primIsLeaf.Begin() + primitiveCount - 1)) ^ 1;
            interiorPrimCount = m_primIsNoLeafScanner.GetPrefixSum()[primitiveCount - 1] + last;
            leafPrimCount = primitiveCount - interiorPrimCount;
            interiorPrimCount = interiorPrimCount > primitiveCount ? 0 : interiorPrimCount;
            
            m_maskedInteriorContent.Resize(nodeCount);
            m_maskedleafContent.Resize(nodeCount);
            m_interiorContentScanner.Resize(nodeCount);
            m_leafContentScanner.Resize(nodeCount);

            createContentMasks<<<nodeGrid, nodeCount>>>(
                m_nodes_IsLeaf.Begin()() + nodeOffset,
                m_nodes_ContentCount.Begin()(), 
                m_maskedleafContent.Begin()(), 
                m_maskedInteriorContent.Begin()(), nodeCount);

            m_interiorContentScanner.ExcScan(m_maskedInteriorContent.Begin(), m_maskedInteriorContent.End(), nutty::PrefixSumOp<CTuint>());
            m_leafContentScanner.ExcScan(m_maskedleafContent.Begin(), m_maskedleafContent.End(), nutty::PrefixSumOp<CTuint>());

            if(lastLeafCount)
            {
                contentStartOffset = m_leafNodesContentCount[lastLeafCount-1];
                contentStartOffset += m_leafNodesContentStart[lastLeafCount-1];
            }

            m_leafNodesContentStart.Resize(currentLeafCount + leafCount);
            m_leafNodesContentCount.Resize(currentLeafCount + leafCount);
            m_leafNodesContent.Resize(leafContentOffset + leafPrimCount);

            m_newNodesContentCount.Resize(m_nodes_ContentCount.Size());
            m_newNodesContentStartAdd.Resize(m_nodes_ContentCount.Size());

            if(d == m_depth-1)
            {
            compactLeafNInteriorData<1><<<nodeGrid, nodeCount>>>(

                m_interiorContentScanner.GetPrefixSum().Begin()(), 
                m_leafContentScanner.GetPrefixSum().Begin()(),
                m_nodes_ContentCount.Begin()(),
                m_nodesBBox[1].Begin()(),
                m_nodes_IsLeaf.Begin()() + nodeOffset,

                m_leafCountScanner.GetPrefixSum().Begin()(), 
                m_interiorCountScanner.GetPrefixSum().Begin()(),
                m_newNodesContentCount.Begin()(),
                m_newNodesContentStartAdd.Begin()(),
                m_leafNodesContentStart.Begin()(), 
                m_leafNodesContentCount.Begin()(), 
                m_nodesBBox[0].Begin()(),

                d, lastLeafCount, contentStartOffset, nodeCount
                );
            }
            else
            {
            compactLeafNInteriorData<0><<<nodeGrid, nodeCount>>>(

                m_interiorContentScanner.GetPrefixSum().Begin()(), 
                m_leafContentScanner.GetPrefixSum().Begin()(),
                m_nodes_ContentCount.Begin()(),
                m_nodesBBox[1].Begin()(),
                m_nodes_IsLeaf.Begin()() + nodeOffset,

                m_leafCountScanner.GetPrefixSum().Begin()(), 
                m_interiorCountScanner.GetPrefixSum().Begin()(),
                m_newNodesContentCount.Begin()(),
                m_newNodesContentStartAdd.Begin()(),
                m_leafNodesContentStart.Begin()(), 
                m_leafNodesContentCount.Begin()(), 
                m_nodesBBox[0].Begin()(),

                d, lastLeafCount, contentStartOffset, nodeCount
                );
            }

            nutty::Copy(m_nodes_ContentCount.Begin(), m_newNodesContentCount.Begin(), m_newNodesContentCount.Begin() + nodeCount);
            nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin() + nodeCount);

            m_newInteriorContent.Resize(interiorPrimCount);
            m_newPrimNodeIndex.Resize(interiorPrimCount);
            m_newPrimPrefixSum.Resize(interiorPrimCount);

            CTuint block = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
            CTuint grid = nutty::cuda::GetCudaGrid(primitiveCount, block);
            PRINT_BUFFER(m_primNodeIndex);
            compactLeafNInteriorContent<<<grid, block>>>
                (
                m_leafNodesContent.Begin()() + leafContentOffset, 
                m_newInteriorContent.Begin()(),
                m_leafCountScanner.GetPrefixSum().Begin()(),
                m_primIsLeaf.Begin()(), 
                m_primIsLeafScanner.GetPrefixSum().Begin()(),
                m_primIsNoLeafScanner.GetPrefixSum().Begin()(),
                m_primIndex.Begin()(), 

                m_primNodeIndex.Begin()(),
                m_primPrefixSum.Begin()(),
                m_newPrimNodeIndex.Begin()(),
                m_newPrimPrefixSum.Begin()(),

                primitiveCount
                );

            nutty::Copy(m_primIndex.Begin(), m_newInteriorContent.Begin(), m_newInteriorContent.Begin() + interiorPrimCount);
            nutty::Copy(m_primNodeIndex.Begin(), m_newPrimNodeIndex.Begin(), m_newPrimNodeIndex.Begin() + interiorPrimCount);
            nutty::Copy(m_primPrefixSum.Begin(), m_newPrimPrefixSum.Begin(), m_newPrimPrefixSum.Begin() + interiorPrimCount);

            leafContentOffset += leafPrimCount;
            primitiveCount = interiorPrimCount;
        }
        else
        {
            nutty::Copy(m_nodesBBox[0].Begin(), m_nodesBBox[1].Begin(), m_nodesBBox[1].Begin() + 2 * levelInteriorNodesCount);
        }

        nodeOffset = currentInteriorNodesCount + currentLeafCount;

        levelInteriorNodesCount = 2 * (levelInteriorNodesCount - leafCount);

        currentInteriorNodesCount += levelInteriorNodesCount;
        
        currentLeafCount += leafCount;
        
        PRINT_BUFFER(m_primNodeIndex);
        PRINT_BUFFER(m_primIndex);
        PRINT_BUFFER(m_nodes_ContentCount);
        PRINT_BUFFER(m_nodes_ContentStartAdd);
        PRINT_BUFFER(m_nodes_Split);
        PRINT_BUFFER(m_nodes_SplitAxis);
        PRINT_BUFFER(m_nodes_IsLeaf);
        PRINT_BUFFER(m_leafNodesContent);
        PRINT_BUFFER(m_leafNodesContentCount);
        PRINT_BUFFER(m_leafNodesContentStart);

        if(interiorPrimCount == 0) //all nodes are leaf nodes
        {
            primitiveCount = lastCnt;
            ct_printf("interiorPrimCount == 0, Bailed out...\n");
            break;
        }
        
        if(d < m_depth-1) //are we not done?
        {
            if(2 * primitiveCount > m_edgeMask.Size())
            {
                GrowMemory();
            }

            if(m_nodes_ContentStartAdd.Size() < levelInteriorNodesCount)
            {
                GrowPerLevelNodeMemory();
            }

            if(m_nodes_IsLeaf.Size() < (currentInteriorNodesCount + 2 * levelInteriorNodesCount))
            {
                GrowNodeMemory();
            }
        }
    }

    m_nodes_NodeIdToLeafIndex.Resize(currentInteriorNodesCount + currentLeafCount);
    nutty::ExclusivePrefixSumScan(m_nodes_IsLeaf.Begin(), m_nodes_IsLeaf.Begin() + currentInteriorNodesCount + currentLeafCount, m_nodes_NodeIdToLeafIndex.Begin(), m_edgeMaskSums.Begin());

    m_interiorNodesCount = currentInteriorNodesCount;
    m_leafNodesCount = currentLeafCount;

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