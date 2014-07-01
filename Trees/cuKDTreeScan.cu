
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#include "cuKDTree.h"
#include "kd_kernel.h"
#include "kd_scan_kernel.h"
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
#include <chimera/Timer.h>

#define PRINT_OUT
#ifndef _DEBUG
#undef PRINT_OUT
#endif

#ifndef PRINT_OUT
#undef PRINT_BUFFER(_name)
#undef PRINT_BUFFER_N(_name, _tmp)
#undef ct_printf

#define PRINT_BUFFER(_name)
#define PRINT_BUFFER_N(_name, _tmp)
#define ct_printf(...)
#endif

#define PREPARE_KERNEL(kernel_name, N) \
    { \
    CTuint block = nutty::cuda::GetCudaBlock(N, 256U); \
    CTuint grid = nutty::cuda::GetCudaGrid(N, block); \

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

void cuKDTreeScan::InitBuffer(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;

    m_depth = (byte)min(64, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));

    m_primAABBs.Resize(primitiveCount); nutty::ZeroMem(m_primAABBs);

    GrowNodeMemory();
    GrowPerLevelNodeMemory(64);
    GrowPrimitiveEventMemory();

    ClearBuffer();
}

void cuKDTreeScan::ClearBuffer(void)
{
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
}

void cuKDTreeScan::GrowPerLevelNodeMemory(CTuint newSize)
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

void cuKDTreeScan::GrowNodeMemory(void)
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

void cuKDTreeScan::GrowPrimitiveEventMemory(void)
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
}

void cuKDTreeScan::PrintStatus(const char* msg /* = NULL */)
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

void cuKDTreeScan::ComputeSAH_Splits(
    CTuint nodeCount,
    CTuint primitiveCount, 
    const CTuint* hNodesContentCount, 
    const CTuint* nodesContentCount)
{
    CTuint elementBlock = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);

    CTuint eventCount = 2 * primitiveCount;
    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    CTuint start = 0;

    m_pool.Reset();
     
    chimera::util::HTimer timer;
    cudaDeviceSynchronize();
    timer.Start();

    /*Event eventsSrc = m_events[0];
    Event eventsDst = m_events[1];

    createEvents<<<elementGrid, elementBlock>>>(eventsSrc, m_primAABBs.Begin()(), m_nodesBBox[0].Begin()(), nodesContent, primitiveCount);

    DEVICE_SYNC_CHECK();

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

    reorderEvents<<<eventGrid, eventBlock>>>(eventsDst, eventsSrc, eventCount);*/

//     m_scannedEventTypeStartMask.Resize(eventCount);
//     m_eventTypeMaskSumsStart.Resize(eventCount);
// 
//     m_scannedEventTypeEndMask.Resize(eventCount);
//     m_eventTypeMaskSumsEnd.Resize(eventCount);
// 
//     nutty::DevicePtr<CTbyte> typeBegin = nutty::DevicePtr_Cast<CTbyte>(m_events3.GetDst().type);
//     nutty::DevicePtr<CTbyte> typeEnd = nutty::DevicePtr_Cast<CTbyte>(m_events3.GetDst().type + eventCount);
// 
//     EventStartScanOp<CTbyte> op0;
//     nutty::ExclusiveScan(typeBegin, typeBegin, m_scannedEventTypeStartMask.Begin(), m_eventTypeMaskSumsStart.Begin(), op0);
// 
//     EventEndScanOp<CTbyte> op1;
//     nutty::ExclusiveScan(typeBegin, typeEnd, m_scannedEventTypeEndMask.Begin(), m_eventTypeMaskSumsEnd.Begin(), op1);

    for(CTbyte i = 0; i < 3; ++i)
    {
        m_events3[i].ScanEventTypes();
    }

    cuEventLineTriple tripleLine(m_events3);

    DEVICE_SYNC_CHECK();

    PRINT_BUFFER(m_nodesBBox[0]);

    computeSAHSplits3<1><<<eventGrid, eventBlock>>>(
        tripleLine,
        nodesContentCount,
        m_splits,
        m_nodesBBox[0].Begin()(),
        eventCount);
    
#if 1
    for(int i = 0; i < eventCount; ++i)
    {
            ct_printf("%d [%d %d] id=%d Axis=%d, Plane=%.4f SAH=%.4f\n", 
                i, m_splits_Below[i], m_splits_Above[i],
                m_splits_IndexedSplit[i].index, 
                (CTuint)m_splits_Axis[i],
                m_splits_Plane[i], 
                m_splits_IndexedSplit[i].sah);
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
        ct_printf("axis=%d plane=%f sah=%f below=%d above=%d\n", (CTuint)axis, plane, s.sah, below, above);

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
    //__ct_printf("|%f ", timer.GetMillis());
    for(CTuint i = 0; i < min(m_pool.GetStreamCount(), nodeCount); ++i)
    {
        nutty::cuStream& stream = m_pool.GetStream(i);
        nutty::cuEvent e = stream.RecordEvent();
        cudaStreamWaitEvent(0, e.GetPointer(), 0);
    }
    
    nutty::SetDefaultStream();
}

CTuint cuKDTreeScan::CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint nodeRange)
{
    m_leafCountScanner.Resize(nodeRange);
    m_leafCountScanner.ExcScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, TypeOp<CTbyte>());

    CTuint leafCount = m_leafCountScanner.GetPrefixSum()[nodeRange-1] + (*(isLeafBegin + nodeOffset + nodeRange - 1) == 1);

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

MakeLeavesResult cuKDTreeScan::MakeLeaves(
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

void EventLine::ScanEvents(CTuint length)
{
    nutty::ExclusivePrefixSumScan(mask.Begin(), mask.Begin() + length, scannedMasks.Begin(), maskSums.Begin());
}

void EventLine::CompactClippedEvents(CTuint length)
{
    CTuint block = nutty::cuda::GetCudaBlock(length, 256U);
    CTuint grid = nutty::cuda::GetCudaGrid(length, block);
    PREPARE_KERNEL(compactEventLine, length)
        compactEventLine<<<grid, block>>>(GetDst(), GetSrc(), mask.Begin()(), scannedMasks.Begin()(), length);
    }
}

void EventLine::ScanEventTypes(void)
{
    EventStartScanOp<CTbyte> op0;
    EventEndScanOp<CTbyte> op1;
    nutty::ExclusiveScan(type.Begin(1), type.Begin(1) + eventCount, scannedEventTypeStartMask.Begin(), eventTypeMaskSumsStart.Begin(), op0);
    nutty::ExclusiveScan(type.Begin(1), type.Begin(1) + eventCount, scannedEventTypeEndMask.Begin(), eventTypeMaskSumsEnd.Begin(), op1);
}

void PrintEventLine(EventLine& line, CTuint l)
{
    PRINT_BUFFER_N(line.indexedEvent[0], l);
    PRINT_BUFFER_N(line.indexedEvent[1], l);
    ct_printf("\n");
    PRINT_BUFFER_N(line.nodeIndex[0], l);
    PRINT_BUFFER_N(line.nodeIndex[1], l);
    ct_printf("\n");
    PRINT_BUFFER_N(line.prefixSum[0], l);
    PRINT_BUFFER_N(line.prefixSum[1], l);
    ct_printf("\n");
    PRINT_BUFFER_N(line.primId[0], l);
    PRINT_BUFFER_N(line.primId[1], l);
    ct_printf("\n");
}

CT_RESULT cuKDTreeScan::Update(void)
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

    //initNodesContent<<<_grid, _block>>>(m_nodesContent, primitiveCount);

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
    
    CTuint elementBlock = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);

    m_events3[0].Resize(2 * primitiveCount);

    m_events3[1].Resize(2 * primitiveCount);

    m_events3[2].Resize(2 * primitiveCount);

    cuEventLineTriple src(m_events3, 0);
    createEvents3<1, 0><<<elementGrid, elementBlock>>>(src, m_primAABBs.Begin()(), m_nodesContent, primitiveCount);

    DEVICE_SYNC_CHECK();

    for(CTbyte i = 0; i < 3; ++i)
    {
        m_events3[i].eventCount = 2 * primitiveCount;
        nutty::Sort(
            nutty::DevicePtr_Cast<IndexedEvent>(m_events3[i].GetSrc().indexedEvent), 
            nutty::DevicePtr_Cast<IndexedEvent>(m_events3[i].GetSrc().indexedEvent + 2 * primitiveCount), 
            EventSort());
    }

    DEVICE_SYNC_CHECK();

    cuEventLineTriple dst(m_events3);

    reorderEvent3<<<2 * elementGrid, elementBlock>>>(src, dst, 2 * primitiveCount);

    DEVICE_SYNC_CHECK();

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
            m_nodes_ContentCount.Begin()());

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

        //todo
        //setEventBelongsToLeaf<<<g, b>>>(m_nodesContent, m_activeNodesIsLeaf.Begin()(), primitiveCount);

        m_newNodesContentCount.Resize(m_nodes_ContentCount.Size());
        m_newNodesContentStartAdd.Resize(m_nodes_ContentCount.Size());

        m_lastNodeContentStartAdd.Resize(m_newNodesContentStartAdd.Size());
        nutty::Copy(m_lastNodeContentStartAdd.Begin(), m_nodes_ContentStartAdd.Begin(), m_nodes_ContentStartAdd.Begin() + nodeCount);

        MakeLeavesResult leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_nodeOffset, 0, nodeCount, primitiveCount, g_currentLeafCount, g_leafContentOffset, 0);

        CTuint lastLeaves = leavesRes.leafCount;
        CTuint lastCnt = primitiveCount;
        primitiveCount = leavesRes.interiorPrimitiveCount;

        for(CTbyte i = 0; i < 3; ++i)
        {
            CTuint count = m_events3[i].GetEventCount();
            CTuint block = nutty::cuda::GetCudaBlock(count, 256U);
            CTuint grid = nutty::cuda::GetCudaGrid(count, b);

            m_events3[i].Resize(2 * count);
            nutty::ZeroMem(m_events3[i].mask);

            clipEvents<<<grid, block>>>(m_events3[i].GetSrc(), m_events3[i].GetDst(), m_events3[i].mask.Begin()(), m_nodes_ContentStartAdd.Begin()(), m_splits, i, count);
            //PrintEventLine(m_events3[i], 2 * count);
            m_events3[i].ScanEvents(2 * count);
            m_events3[i].CompactClippedEvents(2 * count);

            CTuint ccLeft = m_events3[i].scannedMasks[count - 1] + m_events3[i].mask[count - 1];
            CTuint ccRight = m_events3[i].scannedMasks[2 * count - 1] + m_events3[i].mask[2 * count - 1] - ccLeft;
            PRINT_BUFFER(m_events3[i].mask);
            PRINT_BUFFER(m_events3[i].scannedMasks);
            m_events3[i].eventCount = ccRight + ccLeft;
            PrintEventLine(m_events3[i], m_events3[i].eventCount);
        }

        if(leavesRes.interiorPrimitiveCount)
        {
            g_leafContentOffset += leavesRes.leafPrimitiveCount;

            //classifyEvents3<<<eventGrid, eventBlock>>>(m_nodes, m_events[1], m_splits, m_activeNodesIsLeaf.Begin()(), m_edgeMask.Begin()(), m_lastNodeContentStartAdd.Begin()(), eventCount);
            DEVICE_SYNC_CHECK();

            /*
            nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.Begin() + eventCount, m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());
            DEVICE_SYNC_CHECK();

            primitiveCount = m_scannedEdgeMask[eventCount - 1] + m_edgeMask[eventCount - 1];

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

            primitiveCount = leavesRes.interiorPrimitiveCount; */

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
//             if(2 * primitiveCount > m_edgeMask.Size())
//             {
//                 GrowPrimitiveEventMemory();
//             }

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

void cuKDTreeScan::ValidateTree(void)
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