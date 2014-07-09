
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

#define PROFILE

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

#define PREPARE_KERNEL(N) \
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

void PrintEventLine(EventLine& line, CTuint l)
{
    ct_printf("PrintEventLine\n");
//    PRINT_BUFFER_N(line.indexedEvent[line.toggleIndex], l);
    PRINT_BUFFER_N(line.nodeIndex[line.toggleIndex], l);
//     PRINT_BUFFER_N(line.prefixSum[line.toggleIndex], l);
//     PRINT_BUFFER_N(line.primId[line.toggleIndex], l);
//     PRINT_BUFFER_N(line.type[line.toggleIndex], l);
    ct_printf("End\n");
}

void cuKDTreeScan::InitBuffer(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;

    m_depth = (byte)min(64, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));

    m_primAABBs.Resize(primitiveCount); nutty::ZeroMem(m_primAABBs);

    GrowNodeMemory();
    GrowPerLevelNodeMemory(64);
    GrowSplitMemory(4 * primitiveCount);

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

void cuKDTreeScan::GrowSplitMemory(CTuint eventCount)
{
    m_splits_Above.Resize(eventCount);
    m_splits_Below.Resize(eventCount);
    m_splits_Axis.Resize(eventCount);
    m_splits_Plane.Resize(eventCount);
    m_splits_IndexedSplit.Resize(eventCount);

    m_eventIsLeaf.Resize(eventCount);
    
    m_splits.above = m_splits_Above.GetDevicePtr()();
    m_splits.below = m_splits_Below.GetDevicePtr()();
    m_splits.axis = m_splits_Axis.GetDevicePtr()();
    m_splits.indexedSplit = m_splits_IndexedSplit.GetDevicePtr()();
    m_splits.v = m_splits_Plane.GetDevicePtr()();
}

void cuKDTreeScan::PrintStatus(const char* msg /* = NULL */)
{
    ct_printf("PrintStatus: %s\n", msg == NULL ? "" : msg);
    PRINT_BUFFER(m_nodes_ContentCount);
    PRINT_BUFFER(m_nodes_ContentStartAdd);
}

void cuKDTreeScan::ComputeSAH_Splits(
    CTuint nodeCount,
    CTuint eventCount, 
    const CTuint* hNodesContentCount, 
    const CTuint* nodesContentCount)
{
    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    CTuint start = 0;

    m_pool.Reset();
    
    for(CTbyte i = 0; i < 3; ++i)
    {
        m_events3[i].ScanEventTypes();
    }

    cuEventLineTriple tripleLine(m_events3, 0);

    DEVICE_SYNC_CHECK();

    PRINT_BUFFER(m_nodesBBox[0]);

    computeSAHSplits3<1><<<eventGrid, eventBlock>>>(
        tripleLine,
        nodesContentCount,
        m_splits,
        m_nodesBBox[0].Begin()(),
        eventCount);

#if 0
    for(int i = 0; i < eventCount; ++i)
    {
            ct_printf("%d [%d %d] id=%d Axis=%d, Plane=%f SAH=%f :: ", 
                i, m_splits_Below[i], m_splits_Above[i],
                m_splits_IndexedSplit[i].index, 
                (CTuint)m_splits_Axis[i],
                m_splits_Plane[i], 
                (m_splits_IndexedSplit[i].sah == INVALID_SAH ? -1 : m_splits_IndexedSplit[i].sah));

            BBox bbox = m_nodesBBox[0][ m_events3[0].nodeIndex[m_events3[0].toggleIndex][i] ];
            ct_printf("%f %f %f | %f %f %f\n", bbox.m_min.x, bbox.m_min.y, bbox.m_min.z, bbox.m_max.x, bbox.m_max.y, bbox.m_max.z);
    }
#endif

    start = 0;
    m_pool.Reset();

    chimera::util::HTimer timer;
    cudaDeviceSynchronize();
    timer.Start();

    if(nodeCount > 5)
    {
        CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
        CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        dpReduceSAHSplits<<<nodeGrid, nodeBlock>>>(m_nodes, m_splits_IndexedSplit.Begin()(), nodeCount);
//     cudaDeviceSynchronize();
// 
//     for(int i = 0; i < nodeCount; ++i)
//     {
//         CTuint cc = hNodesContentCount[i];
//         CTuint length = 2 * cc;
// 
//         IndexedSAHSplit s = *(m_splits_IndexedSplit.Begin() + start);
//         std::stringstream ss;
//         ss << m_nodesBBox[0][i];
//         __ct_printf("%s ", ss.str().c_str());
//         __ct_printf("id=%d, memoryadd=%d ", s.index, start);
//         CTreal plane = m_splits_Plane[s.index];
//         CTbyte axis = m_splits_Axis[s.index];
//         CTuint below = m_splits_Below[s.index];
//         CTuint above = m_splits_Above[s.index];
//         __ct_printf("axis=%d plane=%f sah=%f below=%d above=%d\n", (CTuint)axis, plane, s.sah, below, above);
// 
//         start += length;
//     }
    }
    else 
    {
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

            if(IS_INVALD_SAH(s.sah))
            {
                for(int i = start; i < start + length; ++i)
                {
                        ct_printf("%d [%d %d] id=%d Axis=%d, Plane=%f SAH=%f :: ", 
                            i, m_splits_Below[i], m_splits_Above[i],
                            m_splits_IndexedSplit[i].index, 
                            (CTuint)m_splits_Axis[i],
                            m_splits_Plane[i], 
                            (m_splits_IndexedSplit[i].sah == INVALID_SAH ? -1 : m_splits_IndexedSplit[i].sah));

                        BBox bbox = m_nodesBBox[0][ m_events3[0].nodeIndex[m_events3[0].toggleIndex][i] ];
                        ct_printf("%f %f %f | %f %f %f\n", bbox.m_min.x, bbox.m_min.y, bbox.m_min.z, bbox.m_max.x, bbox.m_max.y, bbox.m_max.z);
                }
                __debugbreak();
            }
    #endif
            start += length;
        }

        for(CTuint i = 0; i < min(m_pool.GetStreamCount(), nodeCount); ++i)
        {
            nutty::cuStream& stream = m_pool.GetStream(i);
            nutty::cuEvent e = stream.RecordEvent();
            cudaStreamWaitEvent(0, e.GetPointer(), 0);
        }
    
        nutty::SetDefaultStream();
    }

    cudaDeviceSynchronize();
    timer.Stop();
    __ct_printf("%f Events=%d Nodes=%d\n", timer.GetMillis(), eventCount, nodeCount);
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

    m_interiorCountScanned.Resize(nodeRange);
    //m_interiorCountScanner.ExcScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, InvTypeOp<CTbyte>());

    PREPARE_KERNEL(nodeRange)
        makeInverseScan<<<grid, block>>>(m_leafCountScanner.GetPrefixSum().Begin()(), m_interiorCountScanned.Begin()(), nodeRange);
    }

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
    CTuint g_nodeOffset, 
    CTuint nodeOffset, 
    CTuint nodeCount, 
    CTuint eventCount, 
    CTuint currentLeafCount, 
    CTuint leafContentOffset,
    CTuint initNodeToLeafIndex)
{
    CTuint leafCount = CheckRangeForLeavesAndPrepareBuffer(isLeafBegin, nodeOffset, nodeCount);

    if(!leafCount)
    {
        MakeLeavesResult result;
        result.leafCount = 0;
        result.interiorPrimitiveCount = eventCount/2;
        result.leafPrimitiveCount = 0;
        return result;
    }
 
    m_leafNodesContentStart.Resize(currentLeafCount + leafCount);
    m_leafNodesContentCount.Resize(currentLeafCount + leafCount);

    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
    CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

    m_eventIsLeafScanner.Resize(eventCount);
    m_eventIsInteriorScanned.Resize(eventCount);

    m_eventIsLeafScanner.ExcScan(m_eventIsLeaf.Begin(), m_eventIsLeaf.Begin() + eventCount, TypeOp<CTbyte>());
    
    //m_primIsNoLeafScanner.ExcScan(m_primIsLeaf.Begin(), m_primIsLeaf.Begin() + eventCount, InvTypeOp<CTbyte>());

    makeInverseScan<<<eventGrid, eventBlock>>>(m_eventIsLeafScanner.GetPrefixSum().Begin()(), m_eventIsInteriorScanned.Begin()(), eventCount);

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

    cuEventLineTriple eventSrc(m_events3, 0);
    cuEventLineTriple eventDst(m_events3, 1);

    CTbyte last = (*(m_eventIsLeaf.Begin() + eventCount - 1));

    CTuint leafPrimCount = m_eventIsLeafScanner.GetPrefixSum()[eventCount - 1] + last;
    CTuint interiorPrimCount = eventCount/2 - leafPrimCount;
    interiorPrimCount = interiorPrimCount > eventCount/2 ? 0 : interiorPrimCount;

    PRINT_BUFFER(m_nodes_ContentStartAdd);
    //if(leafPrimCount)
    {
        compactInteriorEventData<<<eventGrid, eventBlock>>>(
            eventDst,
            eventSrc,
            m_nodes,
            isLeafBegin() + nodeOffset,
            m_interiorCountScanned.Begin()(),
            m_interiorContentScanner.GetPrefixSum().Begin()(),
            m_leafContentScanner.GetPrefixSum().Begin()(),
            eventCount, leafPrimCount);
    }
    

//     for(int i = 0; i < 3; ++i)
//     PrintEventLine(m_events3[i], eventCount);
//     PRINT_BUFFER(m_activeNodesIsLeaf);
//     PRINT_BUFFER(m_interiorCountScanned);
//     PRINT_BUFFER(m_leafContentScanner.GetPrefixSum());
//     PRINT_BUFFER(m_interiorContentScanner.GetPrefixSum());
    PRINT_BUFFER(m_maskedInteriorContent);
//     for(int i = 0; i < 3; ++i)
//     {
//         m_events3[i].Toggle();
//         PrintEventLine(m_events3[i], eventCount);
//         m_events3[i].Toggle();
//     }

    for(CTbyte i = 0; i < 3; ++i)
    {
        m_events3[i].Toggle();
    }
    
    compactLeafNInteriorData<<<nodeGrid, nodeBlock>>>(
        m_interiorContentScanner.GetPrefixSum().Begin()(), 
        m_leafContentScanner.GetPrefixSum().Begin()(),
        m_nodes_ContentCount.Begin()(),
        m_nodes_ContentStartAdd.Begin()(),
        m_nodesBBox[1].Begin()(),
        isLeafBegin() + nodeOffset,

        m_eventIsLeafScanner.GetPrefixSum().Begin()(),
        m_leafCountScanner.GetPrefixSum().Begin()(), 
        m_interiorCountScanned.Begin()(),
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

    m_leafNodesContent.Resize(leafContentOffset + leafPrimCount);

    for(CTbyte i = 0; i < 3; ++i)
    {
        m_events3[i].eventCount = 2 * interiorPrimCount;
    }

    compactLeafData<<<eventGrid, eventBlock>>>(
        eventSrc,
        m_leafNodesContent.Begin()() + leafContentOffset,
        m_eventIsLeaf.Begin()(),
        m_eventIsLeafScanner.GetPrefixSum().Begin()(),
        eventCount);

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
    PREPARE_KERNEL(length)
        compactEventLine<<<grid, block>>>(GetDst(), GetSrc(), mask.Begin()(), scannedMasks.Begin()(), length);
    }
}

void EventLine::ScanEventTypes(void)
{
    EventStartScanOp<CTbyte> op0;
    EventEndScanOp<CTbyte> op1;
    CTbyte add = toggleIndex;
    nutty::ExclusiveScan(type.Begin(add), type.Begin(add) + eventCount, scannedEventTypeStartMask.Begin(), eventTypeMaskSumsStart.Begin(), op0);
    nutty::ExclusiveScan(type.Begin(add), type.Begin(add) + eventCount, scannedEventTypeEndMask.Begin(), eventTypeMaskSumsEnd.Begin(), op1);
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
    cuEventLineTriple dst(m_events3, 1);
    createEvents3<1, 0><<<elementGrid, elementBlock>>>(src, m_primAABBs.Begin()(), primitiveCount);

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

    reorderEvent3<<<2 * elementGrid, elementBlock>>>(dst, src, 2 * primitiveCount);

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
    CTuint eventCount = 2 * primitiveCount;

    m_events3[0].Toggle();
    m_events3[1].Toggle();
    m_events3[2].Toggle();


#ifdef PROFILE
        chimera::util::HTimer g_timer;
        cudaDeviceSynchronize();
        g_timer.Start();
#endif

    for(CTbyte d = 0; d <= m_depth; ++d)
    {
        ct_printf("\nNew Level=%d (%d)\n", d, m_depth);
        for(int i = 0; i < 3 && d>=6; ++i)
        {
          //  PrintEventLine(m_events3[i], eventCount);
        }

        nutty::ZeroMem(m_activeNodesIsLeaf);

        CTuint nodeCount = g_interiorNodesCountOnThisLevel;
        CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
        CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
        CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

        DEVICE_SYNC_CHECK();

        m_hNodesContentCount.Resize(nodeCount);
        nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);
        
        PRINT_BUFFER_N(m_hNodesContentCount, nodeCount);
//         PRINT_BUFFER_N(m_nodes_ContentStartAdd, nodeCount);
//         PRINT_BUFFER(m_activeNodes);

        m_pool.ClearEvents();
#ifdef PROFILE
        chimera::util::HTimer timer;
        cudaDeviceSynchronize();
        timer.Start();
#endif

        ComputeSAH_Splits(
            nodeCount, 
            eventCount, 
            m_hNodesContentCount.Begin()(),
            m_nodes_ContentCount.Begin()());

#ifdef PROFILE
            cudaDeviceSynchronize();
            timer.Stop();
            time += timer.GetMillis();
            //__ct_printf("%f (%f) primitiveCount=%d nodeCount=%d\n", time, timer.GetMillis(), primitiveCount, nodeCount);
#endif
        makeLeafIfBadSplitOrLessThanMaxElements<<<nodeGrid, nodeBlock>>>(
            m_nodes,
            m_nodes_IsLeaf.Begin()() + g_nodeOffset,
            m_activeNodes.Begin()(),
            m_activeNodesIsLeaf.Begin()(), 
            m_splits,
            d == m_depth-1,
            nodeCount);

//         CTuint b = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
//         CTuint g = nutty::cuda::GetCudaGrid(primitiveCount, b);

        //todo
        //setEventBelongsToLeaf<<<g, b>>>(m_nodesContent, m_activeNodesIsLeaf.Begin()(), primitiveCount);

        m_newNodesContentCount.Resize(m_nodes_ContentCount.Size());
        m_newNodesContentStartAdd.Resize(m_nodes_ContentCount.Size());

        m_lastNodeContentStartAdd.Resize(m_newNodesContentStartAdd.Size());
        nutty::Copy(m_lastNodeContentStartAdd.Begin(), m_nodes_ContentStartAdd.Begin(), m_nodes_ContentStartAdd.Begin() + nodeCount);

        MakeLeavesResult leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_nodeOffset, 0, nodeCount, eventCount, g_currentLeafCount, g_leafContentOffset, 0);

        CTuint lastLeaves = leavesRes.leafCount;
        primitiveCount = leavesRes.interiorPrimitiveCount;
        eventCount = 0;
        assert(!leavesRes.leafCount && "currently not working");
        
        CTuint count = m_events3[0].GetEventCount();
        if(leavesRes.interiorPrimitiveCount)
        {
            for(CTbyte i = 0; i < 3; ++i)
            {
                CTuint block = nutty::cuda::GetCudaBlock(count, 256U);
                CTuint grid = nutty::cuda::GetCudaGrid(count, block);

                m_events3[i].Resize(2 * count);
                nutty::ZeroMem(m_events3[i].mask);

#ifdef _DEBUG
                cudaMemset(m_events3[i].GetDst().indexedEvent, 0, 2 * count * sizeof(IndexedEvent));
                cudaMemset(m_events3[i].GetDst().prefixSum, 0, 2 * count * sizeof(CTuint));
#endif
                clipEvents<<<grid, block>>>(
                    m_events3[i].GetDst(), 
                    m_events3[i].GetSrc(), 
                    m_events3[i].mask.Begin()(), 
                    m_nodes_ContentStartAdd.Begin()(), 
                    m_nodes_ContentCount.Begin()(),
                    m_splits, 
                    i, 
                    count);
            }

            for(CTbyte i = 0; i < 3; ++i)
            {
                m_events3[i].Toggle();
                m_events3[i].ScanEvents(2 * count);
                m_events3[i].CompactClippedEvents(2 * count);
                m_events3[i].Toggle();
            }

            for(CTbyte i = 0; i < 3; ++i)
            {
                CTuint ccLeft = m_events3[i].scannedMasks[count - 1] + m_events3[i].mask[count - 1];
                CTuint ccRight = m_events3[i].scannedMasks[2 * count - 1] + m_events3[i].mask[2 * count - 1] - ccLeft;
                m_events3[i].eventCount = ccRight + ccLeft;
            }


            eventCount = m_events3[0].eventCount;

            g_leafContentOffset += leavesRes.leafPrimitiveCount;

            eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
            eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

            //classifyEvents3<<<eventGrid, eventBlock>>>(m_nodes, m_events[1], m_splits, m_activeNodesIsLeaf.Begin()(), m_edgeMask.Begin()(), m_lastNodeContentStartAdd.Begin()(), eventCount);
            DEVICE_SYNC_CHECK();

            /*
            nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.Begin() + eventCount, m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());
            DEVICE_SYNC_CHECK();

            primitiveCount = m_scannedEdgeMask[eventCount - 1] + m_edgeMask[eventCount - 1];
            */

            if(lastLeaves)
            {
                setActiveNodesMask<1><<<nodeGrid, nodeBlock>>>(
                    m_activeNodesThisLevel.Begin()(), 
                    m_activeNodesIsLeaf.Begin()(), 
                    m_interiorCountScanned.Begin()(),
                    0, 
                    nodeCount);
            }
            else
            {
               setActiveNodesMask<0><<<nodeGrid, nodeBlock>>>(
                    m_activeNodesThisLevel.Begin()(), 
                    m_activeNodesIsLeaf.Begin()(), 
                    m_interiorCountScanned.Begin()(),
                    0, 
                    nodeCount);
            }
            
            CTuint childCount = (nodeCount - leavesRes.leafCount) * 2;
            CTuint thisLevelNodesLeft = nodeCount - leavesRes.leafCount;

            nodeBlock = nutty::cuda::GetCudaBlock(thisLevelNodesLeft, 256U);
            nodeGrid = nutty::cuda::GetCudaGrid(thisLevelNodesLeft, nodeBlock);
            cuEventLineTriple events(m_events3, 0);
            initInteriorNodes<<<nodeGrid, nodeBlock>>>(
                events,
                m_nodes,
                m_splits,

                m_activeNodes.Begin()(),
                m_activeNodesThisLevel.Begin()(),

                m_nodesBBox[0].Begin()(), 
                m_nodesBBox[1].Begin()(), 

                m_newNodesContentCount.Begin()(),
                m_newNodesContentStartAdd.Begin()(),
                m_newActiveNodes.Begin()(),
                m_activeNodesIsLeaf.Begin()() + nodeCount,

                g_childNodeOffset,
                g_nodeOffset,
                thisLevelNodesLeft,
                m_lastNodeContentStartAdd.Begin()(),
                m_depth == d+1);

            nutty::Copy(m_activeNodes.Begin(), m_newActiveNodes.Begin(), m_newActiveNodes.Begin() + childCount);
            nutty::Copy(m_nodes_ContentCount.Begin(), m_newNodesContentCount.Begin(), m_newNodesContentCount.Begin() + childCount);
            nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin(), m_newNodesContentStartAdd.Begin() + childCount);

            setEventsBelongToLeaf<<<eventGrid, eventBlock>>>(
                events,
                m_activeNodesIsLeaf.Begin()() + nodeCount,
                m_eventIsLeaf.Begin()(),
                eventCount);

//             compactPrimitivesFromEvents<<<eventGrid, eventBlock>>>(
//                 m_events[1], 
//                 m_nodes, 
//                 m_nodesContent,
//                 m_leafCountScanner.GetPrefixSum().Begin()(),
//                 m_edgeMask.Begin()(), 
//                 m_scannedEdgeMask.Begin()(),
//                 d, eventCount, lastLeaves);

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

            m_interiorContentScanner.Resize(childCount);
            m_interiorContentScanner.ExcScan(m_nodes_ContentCount.Begin(), m_nodes_ContentCount.Begin() + childCount, nutty::PrefixSumOp<CTuint>());
            setPrefixSumAndContentStart<<<eventGrid, eventBlock>>>(events, m_interiorContentScanner.GetPrefixSum().Begin()(), m_nodes_ContentStartAdd.Begin()(), childCount, eventCount);

            leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_childNodeOffset, nodeCount, childCount, eventCount, g_currentLeafCount + lastLeaves, g_leafContentOffset, 1);

            eventCount = m_events3[0].eventCount;

//             if(d >= 6)
//             {
//                 for(CTbyte i = 0; i < 3; ++i)
//                 {
//                     PrintEventLine(m_events3[i], eventCount);
//                     m_events3[i].Toggle();
//                     PrintEventLine(m_events3[i], eventCount);
//                     m_events3[i].Toggle();
//                 }
//             }

//             if(leavesRes.leafCount)
//             {
//                 for(CTbyte i = 0; i < 3; ++i)
//                 {
//                     m_events3[i].Toggle();
//                 }
//             }

            //PrintStatus();
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
        
        if(eventCount == 0 || g_interiorNodesCountOnThisLevel == 0) //all nodes are leaf nodes
        {
            //primitiveCount = lastCnt;
            break;
        }
        
        if(d < m_depth-1) //are we not done?
        {
            //check if we need more memory
            if(eventCount > m_splits_Above.Size())
            {
                GrowSplitMemory(2 * eventCount);
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

#ifdef PROFILE
    cudaDeviceSynchronize();
    g_timer.Stop();
    __ct_printf("Total: %f, Section: %f\n", g_timer.GetMillis(), time);
#endif

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