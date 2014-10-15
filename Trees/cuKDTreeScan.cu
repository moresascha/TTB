#ifdef _DEBUG
#define NUTTY_DEBUG
#else
#undef NUTTY_DEBUG
#endif

#undef NUTTY_DEBUG

// #include <thrust/sort.h>
// #include <thrust/detail/type_traits.h>

//disable / enable optimization

#undef PROFILE

#define USE_STREAM
#define USE_OPT_RED
#define USE_OPT_SCAN
#define USE_CLIP_MASK
#define USE_HYBRID_REDUCTION
#undef USE_HYBRID_COMPACTION

#if 0
#undef USE_STREAM
#undef USE_OPT_RED
#undef USE_OPT_SCAN
#undef USE_CLIP_MASK
#undef USE_HYBRID_REDUCTION
#endif

#undef FAST_COMPACTION
#define FAST_COMP_PROCS (CTuint)64
#define FAST_COMP_LANE_SIZE 256
#include <cutil_math.h>

#include "cuKDTree.h"

static CTuint hybridChangeDepth = 17;

EventLine::EventLine(void) : toggleIndex(0), nodeIndex(NULL)
{
    rawEvents = new thrust::device_vector<float>();
    eventKeys = new thrust::device_vector<unsigned int>();
}

EventLine::~EventLine(void)
{
    delete rawEvents;
    delete eventKeys;
}

void cuKDTreeScan::SetHCDepth(uint d)
{
    hybridChangeDepth = d;
}

#if 1
#include <cuda_occupancy.h>

#include "kd_scan_kernel.h"
#include "shared_kernel.h"
#include "shared_types.h"
#include <Reduce.h>
#include "nopt_reduce.h"

#include <Sort.h>
#include <Scan.h>
#include <queue>
#include <ForEach.h>
#include <Fill.h>
#include <cuda/Globals.cuh>
#include "buffer_print.h"
#include <chimera/Timer.h>
#include <fstream>

#ifdef FAST_COMPACTION

#define PATH 0
#define FUNCNAME 0
#include "generic_kernel.h"
#undef PATH
#undef FUNCNAME

#define PATH 1
#define FUNCNAME 1
#include "generic_kernel.h"
#undef PATH
#undef FUNCNAME

#define PATH 2
#define FUNCNAME 2
#include "generic_kernel.h"
#undef PATH
#undef FUNCNAME

#define PATH 3
#define FUNCNAME 3
#include "generic_kernel.h"
#undef PATH
#undef FUNCNAME

#define PATH 4
#define FUNCNAME 4
#include "generic_kernel.h"
#endif

#ifdef PROFILE

#define PROFILE_FRAMES 32
#define PROFILE_START chimera::util::HTimer timer; cudaDeviceSynchronize(); timer.Start()
#define PROFILE_END cudaDeviceSynchronize(); timer.Stop(); g_time += timer.GetMillis()

std::map<std::string, double> g_profileTable;

#define KERNEL_PROFILE(_name, _info) \
    { \
        auto it = g_profileTable.find(std::string(#_info));\
        if(it == g_profileTable.end()) { g_profileTable.insert(std::pair<std::string, double>(std::string(#_info), 0)); }\
        chimera::util::HTimer timer;\
        cudaDeviceSynchronize();\
        timer.Start();\
        _name; \
        cudaDeviceSynchronize();\
        timer.Stop();\
        it = g_profileTable.find(std::string(#_info));\
        it->second += timer.GetMillis();\
    }

#else
#define PROFILE_START
#define PROFILE_END

#define KERNEL_PROFILE(_name, _info) _name

#endif

#undef PRINT_OUT
#ifndef _DEBUG
#undef PRINT_OUT
#endif

#ifndef PRINT_OUT
#undef PRINT_BUFFER
#undef PRINT_BUFFER_N
#undef PRINT_RAW_BUFFER
#undef PRINT_RAW_BUFFER_N
#undef ct_printf

#define PRINT_BUFFER(_name)
#define PRINT_BUFFER_N(_name, _tmp)
#define PRINT_RAW_BUFFER(_name)
#define PRINT_RAW_BUFFER_N(_name, _N)
#define ct_printf(...)
#endif

//#define NODES_GROUP_SIZE 128U
#define EVENT_GROUP_SIZE 256U

void SortEvents(EventLines* eventLine);

struct cudaErrorBuffer
{
    CTuint* devMemory;

    cudaErrorBuffer(void)
    {
        cudaMalloc(&devMemory, 4 * sizeof(CTuint));

        CTuint null = 0;
        cudaMemcpy(devMemory, &null, 4, cudaMemcpyHostToDevice);
    }

    bool check(void)
    {
        CTuint hostMemory[4];
        cudaMemcpy(&hostMemory, devMemory, 4 * sizeof(CTuint), cudaMemcpyDeviceToHost);

        if(hostMemory[0])
        {
            __ct_printf("GOT ERROR = %d %d %d %d\n", hostMemory[0], hostMemory[1], hostMemory[2], hostMemory[3]);
            //__debugbreak();
            return true;
        }

        CTuint null = 0;
        cudaMemcpy(devMemory, &null, 4, cudaMemcpyHostToDevice);

        return false;
    }

    ~cudaErrorBuffer(void)
    {
        cudaFree(devMemory);
    }
};

void EventLine::Resize(CTuint size)
{
    if(indexedEvent.Size() >= size)
    {
        return;
    }
    size = (CTuint)(1.2 * size);
    typeStartScanned.Resize(size);
    scannedEventTypeStartMask.Resize(size);
    scannedEventTypeStartMaskSums.Resize(size);
    eventScanner.Resize(size);
    typeStartScanner.Resize(size);
    mask.Resize(size);

    indexedEvent.Resize(size);
    type.Resize(size);
    nodeIndex->Resize(size);
    primId.Resize(size);
    ranges.Resize(size);
}

size_t EventLine::Size(void)
{
    return indexedEvent.Size();
}

cuEventLine EventLine::GetPtr(CTbyte index)
{
    cuEventLine events;
    events.indexedEvent = indexedEvent.Begin(index)();
    events.type = type.Begin(index)();
    events.nodeIndex = nodeIndex->Begin(index)();
    events.primId = primId.Begin(index)();
    events.ranges = ranges.Begin(index)();
    events.mask = mask.GetPointer();

    //events.scannedEventTypeStartMask = typeStartScanner.GetPrefixSum().GetConstPointer();
    events.scannedEventTypeStartMask = typeStartScanned.GetConstPointer();
    //events.scannedEventTypeEndMask = scannedEventTypeEndMask.Begin()();

    return events;
}

cuConstEventLine EventLine::GetConstPtr(CTbyte index)
{
    cuConstEventLine events;
    events.indexedEvent = indexedEvent.Begin(index)();
    events.type = type.Begin(index)();
    events.nodeIndex = nodeIndex->Begin(index)();
    events.primId = primId.Begin(index)();
    events.ranges = ranges.Begin(index)();
    events.mask = mask.GetPointer();

    //events.scannedEventTypeStartMask = typeStartScanner.GetPrefixSum().GetConstPointer();
    events.scannedEventTypeStartMask = typeStartScanned.GetConstPointer();
    //events.scannedEventTypeEndMask = scannedEventTypeEndMask.Begin()();

    return events;
}

template<>
struct ShrdMemory<CTuint3>
{
    __device__ CTuint3* Ptr(void) 
    { 
        extern __device__ __shared__ CTuint3 s_b4[];
        return s_b4;
    }
};

double g_time = 0;

void PrintEventLine(EventLine& line, CTuint l)
{
    ct_printf("PrintEventLine\n");
//    PRINT_BUFFER_N(line.indexedEvent[line.toggleIndex], l);
    ///PRINT_BUFFER_N(line.nodeIndex[line.toggleIndex], l);
//     PRINT_BUFFER_N(line.prefixSum[line.toggleIndex], l);
//     PRINT_BUFFER_N(line.primId[line.toggleIndex], l);
//     PRINT_BUFFER_N(line.type[line.toggleIndex], l);
    ct_printf("End\n");
}

template <typename Operator, typename T>
void ScanTriples(ConstTuple<3, T>& src, Tuple<3, CTuint>& scanned, Tuple<3, CTuint>& sums, CTuint N, Operator op, cudaStream_t pStream)
{
    static const CTuint block = 256;

    ConstTuple<3, CTuint> constSums;
    constSums.ts[0] = sums.ts[0];
    constSums.ts[1] = sums.ts[1];
    constSums.ts[2] = sums.ts[2];

    CTuint grid = nutty::cuda::GetCudaGrid(N, block);

    tripleGroupScan<block><<<grid, block, 0, pStream>>>(
        src, scanned, sums, op,
        N);
    
    DEVICE_SYNC_CHECK();

    CTuint sumsCount = nutty::cuda::GetCudaGrid(N, block);

    if(sumsCount > 1)
    {

        nutty::PrefixSumOp<CTuint> _op;
        completeScan2<1024, 3><<<3, 1024, 0, pStream>>>(constSums, sums, _op, sumsCount);
        DEVICE_SYNC_CHECK();

        spreadScannedSums<<<grid-1, block, 0, pStream>>>(scanned, sums, N);
        DEVICE_SYNC_CHECK();
    }
}

template <typename Operator, typename T>
void ScanBinaryTriples(ConstTuple<3, T>& src, Tuple<3, CTuint>& scanned, Tuple<3, CTuint>& sums, CTuint N, Operator op, cudaStream_t pStream)
{
#ifdef USE_OPT_SCAN
    static const CTuint block = 256;

    ConstTuple<3, uchar4> _src;
    _src.ts[0] = (uchar4*)src.ts[0];
    _src.ts[1] = (uchar4*)src.ts[1];
    _src.ts[2] = (uchar4*)src.ts[2];

    ConstTuple<3, CTuint> constSums;
    constSums.ts[0] = sums.ts[0];
    constSums.ts[1] = sums.ts[1];
    constSums.ts[2] = sums.ts[2];

    Tuple<3, uint4> _scanned;
    _scanned.ts[0] = (uint4*)scanned.ts[0];
    _scanned.ts[1] = (uint4*)scanned.ts[1];
    _scanned.ts[2] = (uint4*)scanned.ts[2];

    const uint BLOCK_SIZE = 256;
    const uint elemsPerBlock = 256; //more than 256?
    const uint elemsPerThread = 4 * elemsPerBlock / BLOCK_SIZE;
    const uint scannedElemsPerBlock = elemsPerThread * BLOCK_SIZE;
    uint grid = nutty::cuda::GetCudaGrid(N, scannedElemsPerBlock);

    binaryTripleGroupScan<block, elemsPerBlock><<<grid, BLOCK_SIZE, 0, pStream>>>(
        _src, _scanned, sums, op,
        N);
    
    DEVICE_SYNC_CHECK();

    if(grid > 1)

    {
#if 1
        nutty::PrefixSumOp<CTuint> _op;
        completeScan2<1024, 3><<<3, 1024, 0, pStream>>>(constSums, sums, _op, grid);

        DEVICE_SYNC_CHECK();
#else
        CTuint shrdStepElemperThread = nutty::cuda::GetCudaGrid(sumsCount, 256U);

        switch(shrdStepElemperThread)
        {
        case  1: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  2: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  3: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  4: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  5: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  6: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  7: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  8: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  9: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 10: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 11: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 12: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        default:   __ct_printf("error\n"); exit(0); break;
        };
#endif

        //spreadScannedSums4<<<grid-1, block, 0, pStream>>>(scanned, sums, N, scannedElemsPerBlock);
        uint elems = (N - BLOCK_SIZE * elemsPerThread + (N%2));
        uint g = nutty::cuda::GetCudaGrid(elems, BLOCK_SIZE);
        spreadScannedSums4t<<<g, BLOCK_SIZE>>>(scanned, sums, elems, scannedElemsPerBlock);
        DEVICE_SYNC_CHECK();
    }
#else
    static const CTuint block = 128;

    ConstTuple<3, CTuint> constSums;
    constSums.ts[0] = sums.ts[0];
    constSums.ts[1] = sums.ts[1];
    constSums.ts[2] = sums.ts[2];

    CTuint grid = nutty::cuda::GetCudaGrid(N, block);
    const uint elemsPerThread = 1;
    const uint scannedElemsPerBlock = elemsPerThread * block;
    grid = nutty::cuda::GetCudaGrid(N, scannedElemsPerBlock);

    binaryTripleGroupScanNoOpt<block><<<grid, block, 0, pStream>>>(
        src, scanned, sums, op,
        N);
    
    DEVICE_SYNC_CHECK();

    CTuint sumsCount = grid;

    if(sumsCount > 1)

    {
#if 1
        nutty::PrefixSumOp<CTuint> _op;
        completeScan2NoOpt<1024, 3><<<3, 1024, 0, pStream>>>(constSums, sums, _op, sumsCount);

        DEVICE_SYNC_CHECK();
#else
        CTuint shrdStepElemperThread = nutty::cuda::GetCudaGrid(sumsCount, 256U);

        switch(shrdStepElemperThread)
        {
        case  1: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  2: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  3: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  4: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  5: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  6: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  7: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  8: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  9: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 10: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 11: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 12: completeScan<256><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        default:   __ct_printf("error\n"); exit(0); break;
        };
#endif

        spreadScannedSums<<<grid-1, block, 0, pStream>>>(scanned, sums, N);

//         const CTuint k = 2;
//         CTuint elems = (N - block * elemsPerThread + (N%2)) / k;
//         grid = nutty::cuda::GetCudaGrid(elems, block);
//         spreadScannedSums2<<<grid, block, 0, pStream>>>(scanned, sums, elems);
        DEVICE_SYNC_CHECK();
    }
#endif
}

void cuKDTreeScan::InitBuffer(void)
{
    CTuint primitiveCount = (CTuint)(m_orginalVertices.Size() / 3);

    m_depth = (byte)min(64, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));

    m_primAABBs.Resize(primitiveCount); nutty::ZeroMem(m_primAABBs);

    for(int i = 0; i < 3; ++i)
    {
       // m_events3[i].SetNodeIndexBuffer(&m_eventNodeIndex);
        m_eventLines.eventLines[i].SetNodeIndexBuffer(&m_eventNodeIndex);
    }

    GrowNodeMemory();
    GrowPerLevelNodeMemory(64);
    GrowSplitMemory(4 * primitiveCount);

    ClearBuffer();

    m_dthAsyncIntCopy.Init(2);
    m_dthAsyncByteCopy.Init(2);

    m_dthAsyncNodesContent.Init(100);

    m_gotLeaves.Resize(1);
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

    CTnodeIsLeaf_t* ptr = m_activeNodesIsLeaf.GetPointer();
    cudaMemcpyToSymbolAsync(d_activeNodesIsLeaf, &ptr, sizeof(CTnodeIsLeaf_t*), 0, cudaMemcpyHostToDevice, m_pStream);

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

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(g_nodes, &m_nodes, sizeof(Node), 0, cudaMemcpyHostToDevice, m_pStream));
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

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(g_nodes, &m_nodes, sizeof(Node), 0, cudaMemcpyHostToDevice, m_pStream));
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

    m_eventNodeIndex.Resize(eventCount);

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(g_splits, &m_splits, sizeof(Split), 0, cudaMemcpyHostToDevice, m_pStream));
    
    SplitConst splitsConst;
    splitsConst.above = m_splits_Above.GetDevicePtr()();
    splitsConst.below = m_splits_Below.GetDevicePtr()();
    splitsConst.axis = m_splits_Axis.GetDevicePtr()();
    splitsConst.indexedSplit = m_splits_IndexedSplit.GetDevicePtr()();
    splitsConst.v = m_splits_Plane.GetDevicePtr()();

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbolAsync(g_splitsConst, &splitsConst, sizeof(SplitConst), 0, cudaMemcpyHostToDevice, m_pStream));
}

void cuKDTreeScan::PrintStatus(const char* msg /* = NULL */)
{
    ct_printf("PrintStatus: %s\n", msg == NULL ? "" : msg);
    PRINT_BUFFER(m_nodes_ContentCount);
    PRINT_BUFFER(m_nodes_ContentStartAdd);
}

void cuKDTreeScan::ScanEventTypesTriples(CTuint eventCount)
{
    CTbyte add = m_eventLines.toggleIndex;

    ConstTuple<3, CTeventType_t> ptr;
    ptr.ts[0] = m_eventLines.eventLines[0].type[add].GetConstPointer();
    ptr.ts[1] = m_eventLines.eventLines[1].type[add].GetConstPointer();
    ptr.ts[2] = m_eventLines.eventLines[2].type[add].GetConstPointer();

    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = m_eventLines.eventLines[0].typeStartScanned.GetPointer();
    ptr1.ts[1] = m_eventLines.eventLines[1].typeStartScanned.GetPointer();
    ptr1.ts[2] = m_eventLines.eventLines[2].typeStartScanned.GetPointer();
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = m_eventLines.eventLines[0].scannedEventTypeStartMaskSums.GetPointer();
    sums.ts[1] = m_eventLines.eventLines[1].scannedEventTypeStartMaskSums.GetPointer();
    sums.ts[2] = m_eventLines.eventLines[2].scannedEventTypeStartMaskSums.GetPointer();

    nutty::PrefixSumOp<CTeventType_t> op;
    ScanBinaryTriples(ptr, ptr1, sums, eventCount, op, m_pStream);
//     PrintBuffer(m_eventLines.eventLines[0].scannedEventTypeStartMaskSums, 7);
//     PrintBuffer(m_eventLines.eventLines[0].typeStartScanned, eventCount);
//     __debugbreak();
}

void cuKDTreeScan::ComputeSAH_Splits(
    CTuint nodeCount,
    CTuint eventCount, 
    const CTuint* nodesContentCount)
{
    CTuint eventBlock = EVENT_GROUP_SIZE;
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
    //cuEventLineTriple tripleLine(m_events3, 0);
    CTuint start = 0;

    //m_pool.Reset();
    //m_typeScanner.Resize(eventCount);
    //m_types3.Resize(eventCount);
    
#if 0
     for(CTbyte i = 0; i < 3; ++i)
     {
//        // m_eventLines.eventLines[i].Resize(eventCount);
         m_eventLines.eventLines[i].ScanEventTypes(eventCount);
//         //PRINT_RAW_BUFFER_N(m_eventLines.eventLines[i].typeStartScanner.GetPrefixSum(), eventCount);
// //         PRINT_RAW_BUFFER(m_events3[i].tmpType);
// //         OutputDebugStringA("\n");
//         //nutty::ZeroMem(m_eventLines.eventLines[i].typeStartScanned);
//        
     }
#endif

    
//     static EventStartScanOp<CTbyte> op0;
//     for(CTbyte k = 0; k < 3; ++k)
//     {
//         groupScan<256U, CTbyte, CTuint, EventStartScanOp<CTbyte>> <<<eventGrid, eventBlock>>>(
//             m_eventLines.eventLines[k].type[m_eventLines.toggleIndex].GetConstPointer(),
//             m_eventLines.eventLines[k].typeStartScanned.GetPointer(), 
//             sums.GetPointer(), 
//             op0, eventCount);
//     }

     //nutty::ZeroMem(m_eventLines.eventLines[0].scannedEventTypeEndMaskSums);

     ScanEventTypesTriples(eventCount);
     DEVICE_SYNC_CHECK();

//      PrintBuffer(m_eventLines.eventLines[0].typeStartScanned);
//      PrintBuffer(m_eventLines.eventLines[1].typeStartScanned);
//      PrintBuffer(m_eventLines.eventLines[2].typeStartScanned);

#if 0
    for(CTbyte i = 0; i < 3; ++i)
    {
        nutty::HostBuffer<CTuint> tmp0(eventCount);
        nutty::HostBuffer<CTuint> tmp1(eventCount);
        nutty::Copy(tmp0.Begin(), m_eventLines.eventLines[i].typeStartScanner.GetPrefixSum().Begin(), m_eventLines.eventLines[i].typeStartScanner.GetPrefixSum().Begin() + eventCount);
        nutty::Copy(tmp1.Begin(), m_eventLines.eventLines[i].typeStartScanned.Begin(), m_eventLines.eventLines[i].typeStartScanned.Begin() + eventCount);
        for(int k = 0; k < eventCount; ++k)
        {
            if(tmp1[k] != tmp0[k])
            {
                __ct_printf("error: %d %d %d %d\n", tmp1[k], tmp0[k], k, i);
                //exit(0);
                    const CTuint block = 512; //nutty::cuda::GetCudaBlock(N, 256U);
                CTuint grid = nutty::cuda::GetCudaGrid(eventCount, block);
                size_t sumSize = (eventCount % nutty::cuda::SCAN_ELEMS_PER_BLOCK) == 0 ? eventCount / nutty::cuda::SCAN_ELEMS_PER_BLOCK : (eventCount / nutty::cuda::SCAN_ELEMS_PER_BLOCK) + 1;
                PRINT_RAW_BUFFER_N(m_eventLines.eventLines[i].scannedEventTypeEndMaskSums, sumSize);
                PRINT_RAW_BUFFER_N(m_eventLines.eventLines[i].typeStartScanner.m_scannedSums, sumSize);
                exit(0);
            }
        }
    }
    #endif

    DEVICE_SYNC_CHECK();
    const CTuint elemsPerThread = 1;
    CTuint N = eventCount;//nutty::cuda::GetCudaGrid(eventCount, elemsPerThread);
    CTuint sahBlock = EVENT_GROUP_SIZE;
    CTuint sahGrid = nutty::cuda::GetCudaGrid(N, sahBlock);

//     computeSAHSplits3<1, elemsPerThread><<<sahGrid, sahBlock, 0, m_pStream>>>(
//         nodesContentCount,
//         m_nodes_ContentStartAdd.GetConstPointer(),
//         m_nodesBBox[0].GetConstPointer(),
//         eventCount,
//         m_eventLines.toggleIndex);

    cudaComputeSAHSplits3(
        nodesContentCount,
        m_nodes_ContentStartAdd.GetConstPointer(),
        m_nodesBBox[0].GetConstPointer(),
        eventCount,
        m_eventLines.toggleIndex,
        sahGrid, sahBlock, m_pStream);
//     computeSAHSplits3Old<<<sahGrid, sahBlock, 0, m_pStream>>>(
//         nodesContentCount,
//         m_nodes_ContentStartAdd.Begin()(),
//         m_nodesBBox[0].Begin()(),
//         eventCount,
//         m_eventLines.toggleIndex);

    DEVICE_SYNC_CHECK();

#if 0
    for(int i = 0; i < eventCount; ++i)
    {
            ct_printf("%d [%d %d] id=%d Axis=%d, Plane=%f SAH=%f :: \n", 
                i, m_splits_Below[i], m_splits_Above[i],
                m_splits_IndexedSplit[i].index, 
                (CTuint)m_splits_Axis[i],
                m_splits_Plane[i], 
                (m_splits_IndexedSplit[i].sah == INVALID_SAH ? -1 : m_splits_IndexedSplit[i].sah));

            //BBox bbox = m_nodesBBox[0][ m_events3[0].nodeIndex[m_events3[0].toggleIndex][i] ];
            //ct_printf("%f %f %f | %f %f %f\n", bbox.m_min.x, bbox.m_min.y, bbox.m_min.z, bbox.m_max.x, bbox.m_max.y, bbox.m_max.z);
    }
#endif

    //start = 0;
    //m_pool.Reset();

        if(nodeCount == 1)
        {
            IndexedSAHSplit neutralSplit;
            neutralSplit.index = 0;
            neutralSplit.sah = FLT_MAX;
#ifdef USE_OPT_RED
            ReduceOpt(m_splits_IndexedSplit.Begin(), m_splits_IndexedSplit.Begin(), eventCount, ReduceIndexedSplit(), neutralSplit, m_pStream);
            //nutty::Reduce(m_splits_IndexedSplit.Begin(), m_splits_IndexedSplit.Begin() + eventCount, ReduceIndexedSplit(), neutralSplit, m_pStream);
#else
            ReduceNoOpt(m_splits_IndexedSplit.Begin(), m_splits_IndexedSplit.Begin(), eventCount, ReduceIndexedSplit(), neutralSplit, m_pStream);
#endif
            DEVICE_SYNC_CHECK();
        }
#ifndef USE_HYBRID_REDUCTION
        else if(true)
#else
        else if(nodeCount < 0)
#endif
        {
        //m_hNodesContentCount.Resize(nodeCount);
        //nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);
        m_dthAsyncNodesContent.WaitForCopy();
        for(CTuint i = 0; i < nodeCount; ++i)
        {
            CTuint cc = m_dthAsyncNodesContent[i];
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

#ifdef USE_OPT_RED
            //nutty::Reduce(m_splits_IndexedSplit.Begin() + start, m_splits_IndexedSplit.Begin() + start + length, ReduceIndexedSplit(), neutralSplit, m_pStream);
            ReduceOpt(m_splits_IndexedSplit.Begin() + start, m_splits_IndexedSplit.Begin() + start, length, ReduceIndexedSplit(), neutralSplit, m_pStream);
#else
            ReduceNoOpt(m_splits_IndexedSplit.Begin() + start, m_splits_IndexedSplit.Begin() + start, length, ReduceIndexedSplit(), neutralSplit, m_pStream);
#endif

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

                        BBox bbox;// = m_nodesBBox[0][ m_events3[0].nodeIndex[m_events3[0].toggleIndex][i] ];
                        ct_printf("%f %f %f | %f %f %f\n", bbox.m_min.x, bbox.m_min.y, bbox.m_min.z, bbox.m_max.x, bbox.m_max.y, bbox.m_max.z);
                }
                __debugbreak();
            }
    #endif
            start += length;
        }
        }
    else
    {
        const CTuint blockSize = 512U;
        CTuint N = nodeCount * blockSize;
        CTuint reduceGrid = nodeCount;//nutty::cuda::GetCudaGrid(N, blockSize);

        //cudaErrorBuffer errorBuffer;
        cudaSegReduce<blockSize>(m_splits_IndexedSplit.GetPointer(), N, eventCount, reduceGrid, blockSize, m_pStream);
        //segReduce<blockSize><<<reduceGrid, blockSize, 0, m_pStream>>>(m_splits_IndexedSplit.GetPointer(), N, eventCount);
        DEVICE_SYNC_CHECK();
    }

//         for(CTuint i = 0; i < min(m_pool.GetStreamCount(), nodeCount); ++i)
//         {
//             nutty::cuStream& stream = m_pool.GetStream(i);
//             nutty::cuEvent e = stream.RecordEvent();
//             cudaStreamWaitEvent(0, e.GetPointer(), 0);
//         }
//     
//         nutty::SetDefaultStream();
}

CTuint cuKDTreeScan::CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTnodeIsLeaf_t>::iterator& isLeafBegin, CTuint nodeOffset, CTuint nodeRange)
{
    m_leafCountScanner.Resize(nodeRange);

    m_leafCountScanner.ExcBinaryScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, IsLeafOP<CTnodeIsLeaf_t>(), m_pStream);

    DEVICE_SYNC_CHECK();

    m_dthAsyncIntCopy.WaitForStream(m_stream);
    m_dthAsyncByteCopy.WaitForStream(m_stream);

#ifndef USE_STREAM
    cudaDeviceSynchronize();
#endif

    m_dthAsyncIntCopy.StartCopy(m_leafCountScanner.GetPrefixSum().GetConstPointer() + nodeRange - 1, 0);
    m_dthAsyncByteCopy.StartCopy(isLeafBegin() + nodeOffset + nodeRange - 1, 0);

    CTuint block = nutty::cuda::GetCudaBlock(nodeRange);
    CTuint grid = nutty::cuda::GetCudaGrid(nodeRange, block);

    DEVICE_SYNC_CHECK();

    if(m_interiorCountScanned.Size() <= nodeRange)
    {
        m_interiorCountScanned.Resize(nodeRange);
        m_maskedInteriorContent.Resize(nodeRange);
        m_interiorContentScanner.Resize(nodeRange);
        m_leafContentScanned.Resize(nodeRange);
    }
// 
//     KERNEL_PROFILE(cudaCreateInteriorContentCountMasks(
//         isLeafBegin() + nodeOffset,
//         m_nodes_ContentCount.Begin()(), 
//         m_maskedInteriorContent.Begin()(), nodeRange, grid, block, m_pStream), CreateInteriorContentCountMasks);

    DEVICE_SYNC_CHECK();
    
    InteriorMaskOp op;
    nutty::PrefixSumOp<CTuint> _op;
    m_interiorContentScanner.ExcScanOPI(m_nodes_ContentCount.Begin(), m_nodes_ContentCount.Begin() + nodeRange, op, m_pStream);

   // __ct_printf("%d\n", m);

    DEVICE_SYNC_CHECK();

    //brauch man nicht...
//     makeOthers<<<grid, block, 0, m_pStream>>>(
// 
//         m_nodes_ContentStartAdd.Begin()(), 
//         m_interiorContentScanner.GetPrefixSum().Begin()(), 
//         m_leafContentScanned.Begin()(), 
//         
//         m_leafCountScanner.GetPrefixSum().Begin()(), 
//         m_interiorCountScanned.Begin()(), 
//         
//         nodeRange);

    DEVICE_SYNC_CHECK();

    m_dthAsyncIntCopy.WaitForCopy();
    m_dthAsyncByteCopy.WaitForCopy();
    CTuint leafCount = m_dthAsyncIntCopy[0] + (m_dthAsyncByteCopy[0] == 1);

    DEVICE_SYNC_CHECK();

    return leafCount;
}

MakeLeavesResult cuKDTreeScan::MakeLeaves(
    nutty::DeviceBuffer<CTnodeIsLeaf_t>::iterator& isLeafBegin,
    CTuint g_nodeOffset, 
    CTuint nodeOffset, 
    CTuint nodeCount, 
    CTuint eventCount, 
    CTuint currentLeafCount, 
    CTuint leafContentOffset,
    CTuint initNodeToLeafIndex,
    CTbyte gotLeaves)
{

    CTuint leafCount = 0;
    if(gotLeaves)
    {
         leafCount = CheckRangeForLeavesAndPrepareBuffer(isLeafBegin, nodeOffset, nodeCount);

         DEVICE_SYNC_CHECK();
    }
     
    if(!leafCount)
    {
        MakeLeavesResult result;
        result.leafCount = 0;
        result.interiorPrimitiveCount = eventCount/2;
        result.leafPrimitiveCount = 0;
        return result;
    }
    
    m_dthAsyncIntCopy.WaitForStream(m_stream);
    m_dthAsyncByteCopy.WaitForStream(m_stream);

#ifndef USE_STREAM
    cudaDeviceSynchronize();
#endif

    //m_dthAsyncIntCopy.StartCopy(m_leafContentScanned.GetConstPointer() + nodeCount - 1, 0);
    m_dthAsyncIntCopy.StartCopy(m_interiorContentScanner.GetPrefixSum().GetConstPointer() + nodeCount - 1, 0);
    m_dthAsyncIntCopy.StartCopy(m_nodes_ContentCount.GetConstPointer() + nodeCount - 1, 1);
    m_dthAsyncByteCopy.StartCopy(m_activeNodesIsLeaf.GetConstPointer() + nodeCount + nodeOffset - 1, 0);
 
    m_leafNodesContentStart.Resize(currentLeafCount + leafCount);
    m_leafNodesContentCount.Resize(currentLeafCount + leafCount);

    const CTuint eventBlock = EVENT_GROUP_SIZE;
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount);
    CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);
    

#if 1
//     m_eventIsLeafScanner.Resize(eventCount);
//     m_eventIsLeafScanner.ExcScan(m_eventIsLeaf.Begin(), m_eventIsLeaf.Begin() + eventCount, TypeOp<CTbyte>());

    m_eventIsLeafScanned.Resize(eventCount);
    m_eventIsLeafScannedSums.Resize(eventCount/256 + 256);

//     binaryGroupScan<256><<<eventGrid, eventBlock, 0, m_pStream>>>(
//         m_eventIsLeaf.GetConstPointer(), m_eventIsLeafScanned.GetPointer(), m_eventIsLeafScannedSums.GetPointer(), TypeOp<CTeventIsLeaf_t>(), eventCount);
#ifndef USE_OPT_SCAN
    cudaBinaryGroupScan<256>(m_eventIsLeaf.GetConstPointer(), 
        m_eventIsLeafScanned.GetPointer(), m_eventIsLeafScannedSums.GetPointer(), 
        TypeOp<CTeventIsLeaf_t>(), eventCount, eventGrid, eventBlock, m_pStream);

    DEVICE_SYNC_CHECK();

    CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);

    if(sumsCount > 1)
    {
        nutty::PrefixSumOp<CTuint> _op;
        //completeScan<256><<<1, 256, 0, m_pStream>>>(m_eventIsLeafScannedSums.GetConstPointer(), m_eventIsLeafScannedSums.GetPointer(), _op, sumsCount);
        cudaCompleteScan<256>(m_eventIsLeafScannedSums.GetConstPointer(), m_eventIsLeafScannedSums.GetPointer(), _op, sumsCount, m_pStream);

        DEVICE_SYNC_CHECK();

        cudaSpreadScannedSumsSingle(
            m_eventIsLeafScanned.GetPointer(), m_eventIsLeafScannedSums.GetConstPointer(), eventCount, eventGrid-1, eventBlock, m_pStream);
    }
#else

    cudaBinaryGroupScan<256>(m_eventIsLeaf.GetConstPointer(), 
        m_eventIsLeafScanned.GetPointer(), m_eventIsLeafScannedSums.GetPointer(), 
        TypeOp<CTeventIsLeaf_t>(), eventCount, eventGrid, eventBlock, m_pStream);

    DEVICE_SYNC_CHECK();

    CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);

    if(sumsCount > 1)
    {
        nutty::PrefixSumOp<CTuint> _op;
        //completeScan<256><<<1, 256, 0, m_pStream>>>(m_eventIsLeafScannedSums.GetConstPointer(), m_eventIsLeafScannedSums.GetPointer(), _op, sumsCount);
        cudaCompleteScan<256>(m_eventIsLeafScannedSums.GetConstPointer(), m_eventIsLeafScannedSums.GetPointer(), _op, sumsCount, m_pStream);

        DEVICE_SYNC_CHECK();

        cudaSpreadScannedSumsSingle(
            m_eventIsLeafScanned.GetPointer(), m_eventIsLeafScannedSums.GetConstPointer(), eventCount, eventGrid-1, eventBlock, m_pStream);
    }

#endif

#endif
    DEVICE_SYNC_CHECK();

    if(m_leafNodesContent.Size() < leafContentOffset + eventCount/2)
    {
        m_leafNodesContent.Resize(leafContentOffset + eventCount/2);
    }

    DEVICE_SYNC_CHECK();

    cudaCompactMakeLeavesData(
        isLeafBegin() + nodeOffset,
        m_nodes_ContentStartAdd.GetPointer(),

        m_eventIsLeafScanned.GetConstPointer(),

        m_nodes_ContentCount.GetPointer(),
        m_eventIsLeaf.GetPointer(),

        m_leafCountScanner.GetPrefixSum().GetConstPointer(), 

        m_activeNodes.GetPointer(),
        m_leafCountScanner.GetPrefixSum().GetConstPointer(),
        m_interiorContentScanner.GetPrefixSum().GetConstPointer(),
        m_nodesBBox[1].GetPointer(),

        m_leafNodesContent.GetPointer(),
        m_nodes_NodeIdToLeafIndex.GetPointer(),
        m_newNodesContentCount.GetPointer(),
        m_newNodesContentStartAdd.GetPointer(),
        m_leafNodesContentStart.GetPointer(),
        m_leafNodesContentCount.GetPointer(),
        m_newActiveNodes.GetPointer(),
        m_nodesBBox[0].GetPointer(),
         
        g_nodeOffset,
        leafContentOffset,
        currentLeafCount,
        nodeCount,
        m_eventLines.toggleIndex,
        eventCount,
        eventGrid, eventBlock, m_pStream);

    DEVICE_SYNC_CHECK();

    m_eventLines.Toggle();

    m_dthAsyncIntCopy.WaitForCopy();
    m_dthAsyncByteCopy.WaitForCopy();

    CTuint copyDistance = nodeCount - leafCount;

    if(copyDistance)
    {
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_nodes_ContentCount.GetPointer(), m_newNodesContentCount.GetPointer(), copyDistance * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_nodes_ContentStartAdd.GetPointer(), m_newNodesContentStartAdd.GetPointer(), copyDistance * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_activeNodes.GetPointer(), m_newActiveNodes.GetPointer(), copyDistance * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
    }

    CTuint interiorPrimCount = m_dthAsyncIntCopy[0] + (m_dthAsyncByteCopy[0] == 0) * m_dthAsyncIntCopy[1];

    //if((int)interiorPrimCount < 0)
    {
        ct_printf("interiorPrimCount = %d %d %d %d\n", interiorPrimCount, m_dthAsyncIntCopy[0], (m_dthAsyncByteCopy[0] == 0), m_dthAsyncIntCopy[1]);

       // __debugbreak();
    }

    CTuint leafPrimCount = eventCount/2 - interiorPrimCount;
    leafPrimCount = leafPrimCount > eventCount/2 ? 0 : leafPrimCount;

    MakeLeavesResult result;
    result.leafCount = leafCount;
    result.interiorPrimitiveCount = interiorPrimCount;
    result.leafPrimitiveCount = leafPrimCount;

    DEVICE_SYNC_CHECK();

    return result;
}

void ClipMask::Resize(size_t size, cudaStream_t pStream)
{
    if(mask[0].Size() >= size) return;
    size = (CTuint)(1.2 * size);
    //mask3.Resize(size);
    mask3Scanner.Resize(size);
    for(int i = 0; i < 3; ++i)
    {
        scannedMask[i].Resize(size);
        scannedSums[i].Resize(size);
        mask[i].Resize(size); 
        newSplits[i].Resize(size);
        index[i].Resize(size);
        elemsPerTile[i].Resize(size);
        scannedOverlappingMasks[i].Resize(size);
        scannedOverlappingSums[i].Resize(size);
//        maskScanner[i].Resize(size);
    }
    cuClipMaskArray mm;
    GetPtr(mm);
    cudaMemcpyToSymbolAsync(g_clipArray, &mm, sizeof(cuClipMaskArray), 0, cudaMemcpyHostToDevice, pStream);

    cuConstClipMask cmss[3];
    GetConstPtr(cmss[0], 0);
    GetConstPtr(cmss[1], 1);
    GetConstPtr(cmss[2], 2);

    cudaMemcpyToSymbolAsync(cms, &cmss, 3 * sizeof(cuConstClipMask), 0, cudaMemcpyHostToDevice, pStream);
}

void EventLine::ScanEvents(CTuint length)
{
    __ct_printf("fatal error: ScanEvents not working\n");
    exit(-1);
    //eventScanner.ExcScan(mask.Begin(), mask.Begin() + length, nutty::PrefixSumOp<CTbyte>());
}

struct ClipMaskPrefixSum3OP
{
    __device__ CTuint3 operator()(CTbyte3 elem)
    {
        CTuint3 v;
        v.x = isSet(elem.x) ? 1 : 0;
        v.y = isSet(elem.y) ? 1 : 0;
        v.z = isSet(elem.z) ? 1 : 0;
        return v;
    }

    __device__ __host__ CTbyte3 GetNeutral(void)
    {
        CTbyte3 v = {0};
        return v;
    }
};

void ClipMask::ScanMasks(CTuint length)
{
//     for(CTbyte i = 0; i < 3; ++i)
//     {
//         maskScanner[i].ExcScan(mask[i].Begin(), mask[i].Begin() + length, ClipMaskPrefixSumOP());
//     }

    //mask3Scanner.ExcScan(mask3.Begin(), mask3.End(), ClipMaskPrefixSum3OP());
}

void EventLine::CompactClippedEvents(CTuint length)
{
//     PREPARE_KERNEL(length)
//         compactEventLine<<<grid, block>>>(GetDst(), GetSrc(), mask.Begin()(), eventScanner.GetPrefixSum().Begin()(), length);
//     }
}

void EventLine::ScanEventTypes(CTuint eventCount)
{
    EventStartScanOp<CTbyte> op0;
    CTbyte add = toggleIndex;
    typeStartScanner.ExcScan(type.Begin(add), type.Begin(add) + eventCount, op0);
}

void EventLines::BindToConstantMemory(cudaStream_t pStream)
{
    cuEventLineTriple src;//(eventLines, 0);
    src.lines[0] = eventLines[0].GetPtr(0);
    src.lines[1] = eventLines[1].GetPtr(0);
    src.lines[2] = eventLines[2].GetPtr(0);


    cuEventLineTriple dst;//(eventLines, 1);
    dst.lines[0] = eventLines[0].GetPtr(1);
    dst.lines[1] = eventLines[1].GetPtr(1);
    dst.lines[2] = eventLines[2].GetPtr(1);

//     cudaMemcpyToSymbol(g_eventTriples, &src, sizeof(cuEventLineTriple));
//     cudaMemcpyToSymbol(g_eventTriples, &dst, sizeof(cuEventLineTriple), sizeof(cuEventLineTriple));

    cudaMemcpyToSymbolAsync(g_eventTriples, &src, sizeof(cuEventLineTriple), 0, cudaMemcpyHostToDevice, pStream);
    cudaMemcpyToSymbolAsync(g_eventTriples, &dst, sizeof(cuEventLineTriple), sizeof(cuEventLineTriple), cudaMemcpyHostToDevice, pStream);

//     cuConstEventLineTriple constSrc;//(eventLines, 0);
//     src.lines[0] = eventLines[0].GetPtr(0);
//     src.lines[1] = eventLines[1].GetPtr(0);
//     src.lines[2] = eventLines[2].GetPtr(0);
// 
//     cudaMemcpyToSymbolAsync(g_eventSrcTriples, &constSrc, sizeof(cuConstEventLineTriple), 0, cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbolAsync(g_eventDstTriples, &dst, sizeof(cuEventLineTriple), 0, cudaMemcpyHostToDevice);
}

// void EventLines::BindToggleIndexToConstantMemory(void)
// {
//     CTbyte dst = ((toggleIndex+1)%2);
//     cudaMemcpyToSymbol(g_eventSrcIndex, &toggleIndex, sizeof(CTbyte));
//     cudaMemcpyToSymbol(g_eventDstIndex, &dst, sizeof(CTbyte));
// }

void cuKDTreeScan::ScanClipMaskTriples(CTuint eventCount)
{
#ifdef FAST_COMPACTION
    ConstTuple<3, CTuint> ptr;
    const CTuint blockSize = FAST_COMP_LANE_SIZE;
    CTuint tilesPerProc = nutty::cuda::GetCudaGrid(eventCount, FAST_COMP_PROCS * FAST_COMP_LANE_SIZE);
    CTuint threads = FAST_COMP_PROCS * FAST_COMP_LANE_SIZE;
    CTuint grid = nutty::cuda::GetCudaGrid(threads, blockSize);
    countValidElementsPerTile2<blockSize><<<grid, blockSize, 0, m_pStream>>>(tilesPerProc, eventCount);

//     PrintBuffer(m_clipsMask.mask[0], 2 * blockSize);
//     PrintBuffer(m_clipsMask.elemsPerTile[0], FAST_COMP_PROCS);
//     PrintBuffer(m_clipsMask.elemsPerTile[1], FAST_COMP_PROCS);
//     PrintBuffer(m_clipsMask.elemsPerTile[2], FAST_COMP_PROCS);

    ptr.ts[0] = (const CTuint*)m_clipsMask.elemsPerTile[0].GetConstPointer();
    ptr.ts[1] = (const CTuint*)m_clipsMask.elemsPerTile[1].GetConstPointer();
    ptr.ts[2] = (const CTuint*)m_clipsMask.elemsPerTile[2].GetConstPointer();
#else
    ConstTuple<3, CTclipMask_t> ptr;
    ptr.ts[0] = m_clipsMask.mask[0].GetConstPointer();
    ptr.ts[1] = m_clipsMask.mask[1].GetConstPointer();
    ptr.ts[2] = m_clipsMask.mask[2].GetConstPointer();
#endif

    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = m_clipsMask.scannedMask[0].GetPointer();
    ptr1.ts[1] = m_clipsMask.scannedMask[1].GetPointer();
    ptr1.ts[2] = m_clipsMask.scannedMask[2].GetPointer();
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = m_clipsMask.scannedSums[0].GetPointer();
    sums.ts[1] = m_clipsMask.scannedSums[1].GetPointer();
    sums.ts[2] = m_clipsMask.scannedSums[2].GetPointer();

#ifndef FAST_COMPACTION
    ClipMaskPrefixSumOP op;
    ScanBinaryTriples(ptr, ptr1, sums, eventCount, op, m_pStream);

//     ClipMaskIsOverlappingOP __op;
//     ptr1.ts[0] = m_clipsMask.scannedOverlappingMasks[0].GetPointer();
//     ptr1.ts[1] = m_clipsMask.scannedOverlappingMasks[1].GetPointer();
//     ptr1.ts[2] = m_clipsMask.scannedOverlappingMasks[2].GetPointer();
//     sums.ts[0] = m_clipsMask.scannedOverlappingSums[0].GetPointer();
//     sums.ts[1] = m_clipsMask.scannedOverlappingSums[1].GetPointer();
//     sums.ts[2] = m_clipsMask.scannedOverlappingSums[2].GetPointer();
//     ScanTriples(ptr, ptr1, sums, eventCount, __op, m_pStream);
//     //PrintBuffer(m_clipsMask.scannedOverlappingMasks[0], eventCount);
//     __ct_printf("scannedOverlappingMasks x: %d \n", m_clipsMask.scannedOverlappingMasks[0][eventCount-1]);
//     __ct_printf("scannedOverlappingMasks y: %d \n", m_clipsMask.scannedOverlappingMasks[1][eventCount-1]);
//     __ct_printf("scannedOverlappingMasks z: %d \n\n", m_clipsMask.scannedOverlappingMasks[2][eventCount-1]);
#else
    nutty::PrefixSumOp<CTuint> op;
    ScanTriples(ptr, ptr1, sums, FAST_COMP_PROCS, op, m_pStream);

//     cudaDeviceSynchronize();
    /*
    for(int i = 0; i < 3; ++i)
    {
        int cpuSum = 0;
        for(int p = 0; p < eventCount; ++p)
        {
            cpuSum += m_clipsMask.mask[i][p] > 0;
        }

        int sum = 0;
        for(int g = 0; g < warps; ++g)
        {
            sum += m_clipsMask.elemsPerTile[i][g];
        }

        ct_printf("cpuSum=%d sum=%d, %d + %d = %d\n", 
      cpuSum, sum, m_clipsMask.scannedMask[i][warps-1],  m_clipsMask.elemsPerTile[i][warps-1], m_clipsMask.scannedMask[i][warps-1] +  m_clipsMask.elemsPerTile[i][warps-1]);
    }

    PrintBuffer(m_clipsMask.elemsPerTile[0], warps);
    PrintBuffer(m_clipsMask.elemsPerTile[1], warps);
    PrintBuffer(m_clipsMask.elemsPerTile[2], warps);
    */
//     PrintBuffer(m_clipsMask.scannedMask[0], FAST_COMP_PROCS);
//     PrintBuffer(m_clipsMask.scannedMask[1], FAST_COMP_PROCS);
//     PrintBuffer(m_clipsMask.scannedMask[2], FAST_COMP_PROCS);

#endif
    //m_clipsMask.maskScanner[0].ExcScan(m_clipsMask.mask[0].Begin(), m_clipsMask.mask[0].Begin() + eventCount, op, m_pStream);
//     m_clipsMask.maskScanner[1].ExcScan(m_clipsMask.mask[1].Begin(), m_clipsMask.mask[1].Begin() + eventCount, op, m_pStream);
//     m_clipsMask.maskScanner[2].ExcScan(m_clipsMask.mask[2].Begin(), m_clipsMask.mask[2].Begin() + eventCount, op, m_pStream);
}

CT_RESULT cuKDTreeScan::Update(void)
{
    float* rawEvents[3];
    unsigned int* rawEventkeys[3];
    CTuint primitiveCount = (CTuint)(m_currentTransformedVertices.Size() / 3);

    if(!m_initialized)
    {
        InitBuffer();
        m_initialized = true;
        //CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
    }

    for(CTbyte i = 0; i < 3; ++i)
    {
        m_eventLines.eventLines[i].rawEvents->resize(2 * primitiveCount);
        m_eventLines.eventLines[i].eventKeys->resize(2 * primitiveCount);

        rawEventkeys[i] = m_eventLines.eventLines[i].eventKeys->data().get();
        rawEvents[i] = m_eventLines.eventLines[i].rawEvents->data().get();
    }

    //ClearBuffer();

//    static bool staticc = true;
    KERNEL_PROFILE(cudaCreateTriangleAABBs(m_currentTransformedVertices.GetPointer(), m_primAABBs.GetPointer(), primitiveCount, m_pStream), init);
    //PrintBuffer(m_primAABBs);

   // if(staticc)
    {
        DEVICE_SYNC_CHECK();

        static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
        static float3 min3f = -max3f;

        BBox bboxN;
        bboxN.m_min = max3f; 
        bboxN.m_max = min3f;
        m_sceneBBox.Resize(m_primAABBs.Size()/2);
        
        nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), ReduceBBox(), bboxN, m_pStream);

        //staticc = false;
    }

    DEVICE_SYNC_CHECK(); 

    CTuint elementBlock = nutty::cuda::GetCudaBlock(primitiveCount);
    CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);
    
    m_eventLines.Resize(2 * primitiveCount, m_pStream);

#ifdef PROFILE
        chimera::util::HTimer g_timer;
        cudaDeviceSynchronize();
        g_timer.Start();
        g_time = 0;
#endif

    m_eventLines.toggleIndex = 0;

    KERNEL_PROFILE(
        cudaCreateEventsAndInit3(
        m_primAABBs.GetConstPointer(), 
        m_sceneBBox.GetConstPointer(),

        m_activeNodes.GetPointer(),
        m_nodes_NodeIdToLeafIndex.GetPointer(),
        m_nodes_IsLeaf.GetPointer(),
        m_nodes_ContentCount.GetPointer(),
        m_nodesBBox[0].GetPointer(),

        rawEvents,
        rawEventkeys,

        primitiveCount, elementGrid, elementBlock, m_pStream)
        ,init
        );
    DEVICE_SYNC_CHECK();

    KERNEL_PROFILE(SortEvents(&m_eventLines), init);
    DEVICE_SYNC_CHECK();

//     for(CTbyte i = 0; i < 3; ++i)
//     {
//         nutty::Sort(
//             nutty::DevicePtr_Cast<IndexedEvent>(m_eventLines.eventLines[i].GetPtr(0).indexedEvent), 
//             nutty::DevicePtr_Cast<IndexedEvent>(m_eventLines.eventLines[i].GetPtr(0).indexedEvent + 2 * primitiveCount), 
//             EventSort(),
//             m_pStream);
//     }

//     thrust::host_vector<float> h_vec(primitiveCount *2);
//     thrust::host_vector<unsigned int> h_vecK(primitiveCount *2);
//     h_vec = *m_eventLines.eventLines[2].rawEvents;
//     h_vecK = *m_eventLines.eventLines[2].eventKeys;
//     for(int i = 0; i < primitiveCount *2; ++i)
//     {
//         __ct_printf("%f %d -- ", h_vec[i], h_vecK[i]);
//     }
//     __ct_printf("\n\n");
//     PrintBuffer(m_eventLines.eventLines[2].indexedEvent[0]);

    //reorderEvent3<<<2 * elementGrid, elementBlock, 0, m_pStream>>>(2 * primitiveCount);
    KERNEL_PROFILE(cudaReorderEvent3(2 * primitiveCount, 2 * elementGrid, elementBlock, rawEvents, rawEventkeys, m_pStream), init);

    DEVICE_SYNC_CHECK();

    CTuint g_interiorNodesCountOnThisLevel = 1;
    CTuint g_currentInteriorNodesCount = 1;
    CTuint g_currentLeafCount = 0;
    CTuint g_leafContentOffset = 0;
    CTuint g_childNodeOffset = 1;
    CTuint g_nodeOffset = 0;
    CTuint g_entries = 1;
    
    CTuint eventCount = 2 * primitiveCount;
    //CTuint hybridChangeDepth = (3 * m_depth) / 4; //log((2*eventCount)/3);

    m_eventLines.Toggle();
    CTuint eventSum = 0;

    for(CTbyte d = 0; d < m_depth; ++d)
    {
//        static int i = 0;
        //__ct_printf("New Level=%d Events=%d (Frame=%d)\n", d, eventCount, ++i);
        const static CTbyte null = 0;
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_gotLeaves.GetPointer(), &null, sizeof(CTbyte), cudaMemcpyHostToDevice, m_pStream));
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyToSymbolAsync(d_nodeOffset, &g_interiorNodesCountOnThisLevel, sizeof(CTuint), 0, cudaMemcpyHostToDevice, m_pStream));

        CTuint nodeCount = g_interiorNodesCountOnThisLevel;
        CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount);
        CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        CTuint eventBlock = EVENT_GROUP_SIZE;//nutty::cuda::GetCudaBlock(eventCount, 256U);
        CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

        DEVICE_SYNC_CHECK();

#ifndef USE_HYBRID_REDUCTION
        m_dthAsyncNodesContent.WaitForStream(m_stream);
        m_dthAsyncNodesContent.Resize(nodeCount);
        m_dthAsyncNodesContent.StartCopy(m_nodes_ContentCount.GetConstPointer(), 0, nodeCount);
#endif

#if 0
        m_hNodesContentCount.Resize(nodeCount);
        nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);
     //   PrintBuffer(m_hNodesContentCount, nodeCount);
//         for(int i = 0; i < nodeCount; ++i)
//         {
//             if(m_hNodesContentCount[i] > 500000 || m_hNodesContentCount[i] <= MAX_ELEMENTS_PER_LEAF)
//             {
//                 exit(0);
//             }
//         }

        //PrintBuffer(m_nodes_ContentCount, nodeCount);

        PRINT_BUFFER_N(m_nodes_ContentCount, nodeCount);
#endif

        //m_pool.ClearEvents();
        KERNEL_PROFILE(
        ComputeSAH_Splits(
            nodeCount, 
            eventCount,
            m_nodes_ContentCount.Begin()()), ComputeSAH_Splits);

        DEVICE_SYNC_CHECK();

#if 0
        //grad nicht nötig...
        makeLeafIfBadSplitOrLessThanMaxElements<<<nodeGrid, nodeBlock, 0, m_pStream>>>(
            m_nodes,
            m_nodes_IsLeaf.GetPointer() + g_nodeOffset,
            m_activeNodes.GetPointer(),
            m_activeNodesIsLeaf.GetPointer(), 
            m_splits,
            d == m_depth-1,
            nodeCount);

        DEVICE_SYNC_CHECK();
#endif

        m_newNodesContentCount.Resize(m_nodes_ContentCount.Size());
        m_newNodesContentStartAdd.Resize(m_nodes_ContentCount.Size());

        m_lastNodeContentStartAdd.Resize(m_newNodesContentStartAdd.Size());

        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_lastNodeContentStartAdd.GetPointer(), m_nodes_ContentStartAdd.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));

        MakeLeavesResult leavesRes; // = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_nodeOffset, 0, nodeCount, eventCount, g_currentLeafCount, g_leafContentOffset, 0);
        leavesRes.leafCount = 0;
        leavesRes.interiorPrimitiveCount = eventCount/2;
        leavesRes.leafPrimitiveCount = 0;

        CTuint lastLeaves = leavesRes.leafCount;
        primitiveCount = leavesRes.interiorPrimitiveCount;

        if(leavesRes.leafCount) //assert(!leavesRes.leafCount && "currently not working");
        {
            OutputDebugStringA("leavesRes.leafCount currently not working\n");
            exit(0);
        }
        
        DEVICE_SYNC_CHECK();
           
        CTuint count = eventCount;
        
        if(leavesRes.interiorPrimitiveCount)
        {
            CTuint block = EVENT_GROUP_SIZE;
            CTuint grid = nutty::cuda::GetCudaGrid(count, block);    

            m_eventLines.Resize(2 * count, m_pStream);
            m_clipsMask.Resize(2 * count, m_pStream);

//             nutty::ZeroMem(m_clipsMask.mask[0]);
//             nutty::ZeroMem(m_clipsMask.mask[1]);
//             nutty::ZeroMem(m_clipsMask.mask[2]);

//             nutty::ZeroMem(m_clipsMask.maskScanner[0].GetPrefixSum());
//             nutty::ZeroMem(m_clipsMask.maskScanner[1].GetPrefixSum());
//             nutty::ZeroMem(m_clipsMask.maskScanner[2].GetPrefixSum());

//             CTuint tb = 32;
//             CTuint tg = nutty::cuda::GetCudaGrid(count, tb);

            KERNEL_PROFILE(cudaCreateClipMask(
                m_nodes_ContentStartAdd.GetPointer(), 
                m_nodes_ContentCount.GetPointer(),
                count,
                m_eventLines.toggleIndex, grid, block, m_pStream), ClippingAndPartitioning);

#ifndef USE_CLIP_MASK
            KERNEL_PROFILE(cudaClipEventsMask(m_nodes_ContentStartAdd.GetPointer(), 
                m_nodes_ContentCount.GetPointer(),
                count,
                m_eventLines.toggleIndex, grid, block, m_pStream), ClippingAndPartitioning);
#endif

//             std::ofstream file("mask.txt", std::ios::app);
// 
//             for(int axis = 0; axis < 3; ++axis)
//             {
//                 nutty::HostBuffer<CTuint> tmp(2 * count);
//                 nutty::Copy(tmp.Begin(), m_clipsMask.mask[axis].Begin(), m_clipsMask.mask[axis].Begin() + 2 * count);
//                 for(int i = 0; i < 2 * count; ++i)
//                 {
//                     file << (int)tmp[i] << " ";
//                 }
//                 file << "NA ";
//             }
            //file << "NL\n";
// 
//             createClipMask<<<grid, block, 0, m_pStream>>>(
//                 m_nodes_ContentStartAdd.GetPointer(), 
//                 m_nodes_ContentCount.GetPointer(),
//                 count,
//                 m_eventLines.toggleIndex);

//             clipEvents3<<<grid, block, 0, m_pStream>>>(
//                 m_nodes_ContentStartAdd.GetPointer(), 
//                 m_nodes_ContentCount.GetPointer(),
//                 count,
//                 m_eventLines.toggleIndex);



//            CTuint toggleSave = m_eventLines.toggleIndex;

// CTuint prefixSums[3];
// for(int k = 0; k < 3; ++k)
// {
//     nutty::HostBuffer<CTuint> srcEventScan(2 * count);
//     nutty::Copy(srcEventScan.Begin(), m_clipsMask.mask[k].Begin(), m_clipsMask.mask[k].Begin() + 2 * count);
//     prefixSums[k] = 0;
//     for(int i = 0; i < srcEventScan.Size(); ++i)
//     {
//         prefixSums[k] += srcEventScan[i] > 0;
//     }
// }
            DEVICE_SYNC_CHECK();

            //m_clipsMask.ScanMasks(2 * count);

            KERNEL_PROFILE(ScanClipMaskTriples(2 * count), ClippingAndPartitioning);
            //m_clipsMask.mask3Scanner.ExcScan(m_clipsMask.mask3.Begin(), m_clipsMask.mask3.Begin() + 2 * count, ClipMaskPrefixSum3OP());

            DEVICE_SYNC_CHECK();

#ifndef USE_STREAM
            cudaDeviceSynchronize();
#endif
            m_dthAsyncIntCopy.WaitForStream(m_stream);

#ifndef FAST_COMPACTION
            //m_dthAsyncByteCopy.WaitForStream(m_stream);
            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.scannedMask[0].GetConstPointer() + 2 * count - 1), 1);
            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.mask[0].GetPointer() + 2 * count - 1), 0);
#else
            const CTuint blockSize = FAST_COMP_LANE_SIZE;

            CTuint threads = FAST_COMP_PROCS * FAST_COMP_LANE_SIZE;
            CTuint tilesPerProc = nutty::cuda::GetCudaGrid(2 * count, threads);
            CTuint activeWarps =  nutty::cuda::GetCudaGrid(2 * count, tilesPerProc * FAST_COMP_LANE_SIZE);
            CTuint _grid = nutty::cuda::GetCudaGrid(threads, blockSize);
            //int tiles = 2*eventCount / TILE_SIZE + (2*eventCount % TILE_SIZE == 0 ? 0 : 1);
            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.scannedMask[0].GetConstPointer() + min(activeWarps, FAST_COMP_PROCS) - 1), 1);
            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.elemsPerTile[0].GetConstPointer() + min(activeWarps, FAST_COMP_PROCS) - 1), 0);
#endif

            CTuint childCount = (nodeCount - leavesRes.leafCount) * 2;
            CTuint thisLevelNodesLeft = nodeCount - leavesRes.leafCount;
#ifndef FAST_COMPACTION
            CTuint _block = EVENT_GROUP_SIZE;
            CTuint _grid = nutty::cuda::GetCudaGrid(3 * 2 * count, block);

//             compactEventLineV2<<<_grid, _block, 0, m_pStream>>>(
//                 2 * count,
//                 m_eventLines.toggleIndex);
//             chimera::util::HTimer tt;
// 
//             cudaDeviceSynchronize();
//             tt.Start();

            nodeBlock = nutty::cuda::GetCudaBlock(thisLevelNodesLeft);
            nodeGrid = nutty::cuda::GetCudaGrid(thisLevelNodesLeft, nodeBlock);

//         if(nodeCount > 1024)
//         {
//             KERNEL_PROFILE(cudaCompactEventLineV2Buffered(2 * count, m_eventLines.toggleIndex, _grid, m_pStream), tb);
// 
//         }
//         else
//         {
//                         _grid = nutty::cuda::GetCudaGrid(3 * 2 * count, block);
//                     KERNEL_PROFILE(cudaCompactEventLineV2(3 * 2 * count, m_eventLines.toggleIndex, _grid, _block, m_pStream), ta);
//         }
            //if(nodeCount > 15)
            {
#ifndef USE_HYBRID_COMPACTION
                CTuint N = 3 * 2 * count;
                _grid = nutty::cuda::GetCudaGrid(N, block);
                KERNEL_PROFILE(cudaCompactEventLineV2(N, m_eventLines.toggleIndex, _grid, _block, m_pStream), ClippingAndPartitioning);
#else
                if(d >= hybridChangeDepth)
                {
                    __ct_printf("%d %d\n", d, hybridChangeDepth);
                    KERNEL_PROFILE(cudaCompactEventLineV2Buffered(2 * count, m_eventLines.toggleIndex, _grid, m_pStream), ClippingAndPartitioning);
                }
                else
                {
                    CTuint N = 3 * 2 * count;
                    _grid = nutty::cuda::GetCudaGrid(N, block);
                    KERNEL_PROFILE(cudaCompactEventLineV2(N, m_eventLines.toggleIndex, _grid, _block, m_pStream), ClippingAndPartitioning);
                }
#endif
            }
            //else
            {
//                 _grid = nutty::cuda::GetCudaGrid(2 * count, block);
//                 KERNEL_PROFILE(cudaCompactEventLineV2Buffered(2 * count, m_eventLines.toggleIndex, _grid, _block, m_pStream), cudaCompactEventLineV21);
            }

/*            PrintBuffer(m_clipsMask.mask[0], 2 * count);*/
//             Tuple<3, CTuint> ptr1;
//             ptr1.ts[0] = m_clipsMask.scannedOverlappingMasks[0].GetPointer();
//             ptr1.ts[1] = m_clipsMask.scannedOverlappingMasks[1].GetPointer();
//             ptr1.ts[2] = m_clipsMask.scannedOverlappingMasks[2].GetPointer();
//             CTuint b = EVENT_GROUP_SIZE;
// 
//             CTuint g = nutty::cuda::GetCudaGrid(count, block);
//             insertClippedEventsSplitAxis<EVENT_GROUP_SIZE><<<g, b, 0, m_pStream>>>(ptr1, m_nodes_ContentStartAdd.GetConstPointer(), m_eventLines.toggleIndex, count);


            DEVICE_SYNC_CHECK();
//             cudaDeviceSynchronize();
//             tt.Stop();
/*            __ct_printf("%f (%d)\n", tt.GetMillis(), d);*/
//             nutty::cuEvent e = m_stream.RecordEvent();
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[0].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[1].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[2].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[3].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[4].GetPointer(), e.GetPointer(), 0));
//             cuEventDestroy(e.Free());
// 
//             optimizedcompactEventLineV3Type0<<<_grid, _block, 0, m_compactStreams[0].GetPointer()>>>(m_eventLines.toggleIndex, 2 * count);
//             optimizedcompactEventLineV3Type1<<<_grid, _block, 0, m_compactStreams[1].GetPointer()>>>(m_eventLines.toggleIndex, 2 * count);
//             optimizedcompactEventLineV3Type2<<<_grid, _block, 0, m_compactStreams[2].GetPointer()>>>(m_eventLines.toggleIndex, 2 * count);
//             optimizedcompactEventLineV3Type3<<<_grid, _block, 0, m_compactStreams[3].GetPointer()>>>(m_eventLines.toggleIndex, 2 * count);
//             optimizedcompactEventLineV3Type4<<<_grid, _block, 0, m_compactStreams[4].GetPointer()>>>(m_eventLines.toggleIndex, 2 * count);
//             DEVICE_SYNC_CHECK();
// 
//             m_stream.WaitEvent(m_compactStreams[0].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[1].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[2].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[3].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[4].RecordEvent());

#else

            //optimizedcompactEventLineV2<blockSize/32><<<_grid, blockSize>>>(m_eventLines.toggleIndex, tilesPerProc, activeWarps, 2 * count);

            //optimizedcompactEventLineV3<blockSize><<<_grid, blockSize>>>(m_eventLines.toggleIndex, tilesPerProc, activeWarps, 2 * count);
//             nutty::cuEvent e = m_stream.RecordEvent();
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[0].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[1].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[2].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[3].GetPointer(), e.GetPointer(), 0));
//             CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamWaitEvent(m_compactStreams[4].GetPointer(), e.GetPointer(), 0));
//             cuEventDestroy(e.Free());
// 
//             optimizedcompactEventLineV3Type0<blockSize><<<_grid, blockSize, 0, m_compactStreams[0].GetPointer()>>>(m_eventLines.toggleIndex, tilesPerProc, activeWarps, 2 * count);
//             optimizedcompactEventLineV3Type1<blockSize><<<_grid, blockSize, 0, m_compactStreams[1].GetPointer()>>>(m_eventLines.toggleIndex, tilesPerProc, activeWarps, 2 * count);
//             optimizedcompactEventLineV3Type2<blockSize><<<_grid, blockSize, 0, m_compactStreams[2].GetPointer()>>>(m_eventLines.toggleIndex, tilesPerProc, activeWarps, 2 * count);
//             optimizedcompactEventLineV3Type3<blockSize><<<_grid, blockSize, 0, m_compactStreams[3].GetPointer()>>>(m_eventLines.toggleIndex, tilesPerProc, activeWarps, 2 * count);
//             optimizedcompactEventLineV3Type4<blockSize><<<_grid, blockSize, 0, m_compactStreams[4].GetPointer()>>>(m_eventLines.toggleIndex, tilesPerProc, activeWarps, 2 * count);
//             DEVICE_SYNC_CHECK();
// 
//             m_stream.WaitEvent(m_compactStreams[0].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[1].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[2].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[3].RecordEvent());
//             m_stream.WaitEvent(m_compactStreams[4].RecordEvent());
#endif
            m_eventLines.Toggle();

            g_leafContentOffset += leavesRes.leafPrimitiveCount;

//             if(lastLeaves)
//             {
// //                 setActiveNodesMask<1><<<nodeGrid, nodeBlock, 0, m_pStream>>>(
// //                     m_activeNodesThisLevel.Begin()(), 
// //                     m_activeNodesIsLeaf.Begin()(), 
// //                     m_interiorCountScanned.Begin()(),
// //                     0, 
// //                     nodeCount);
//                 KERNEL_PROFILE(cudaSetActiveNodesMask(
//                     m_activeNodesThisLevel.Begin()(), 
//                     m_activeNodesIsLeaf.Begin()(), 
//                     m_interiorCountScanned.Begin()(),
//                     0, 
//                     nodeCount, nodeGrid, nodeBlock, m_pStream), cudaSetActiveNodesMask);
//             }

            //merge into compact events?
            KERNEL_PROFILE(cudaInitInteriorNodes(
                m_activeNodes.GetConstPointer(),
                m_activeNodesThisLevel.GetConstPointer(),

                m_nodesBBox[0].GetConstPointer(), 
                m_nodesBBox[1].GetPointer(), 

                m_nodes_ContentCount.GetPointer(),

                m_newNodesContentCount.GetPointer(),
                m_newActiveNodes.GetPointer(),
                m_activeNodesIsLeaf.GetPointer() + nodeCount,

                g_childNodeOffset,
                g_nodeOffset,
                thisLevelNodesLeft,
                m_lastNodeContentStartAdd.GetPointer(),
                m_gotLeaves.GetPointer(),
                m_depth == d+1,
                leavesRes.leafCount,
                nodeGrid, nodeBlock, m_pStream), CreateChildNodes);

            CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_activeNodes.GetPointer(), m_newActiveNodes.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
            CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_nodes_ContentCount.GetPointer(), m_newNodesContentCount.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));

            DEVICE_SYNC_CHECK();

            m_dthAsyncIntCopy.WaitForCopy();

// #ifndef FAST_COMPACTION
//             m_dthAsyncByteCopy.WaitForCopy();
// #endif

            eventCount = m_dthAsyncIntCopy[1] + 
 #ifndef FAST_COMPACTION
                isSet(m_dthAsyncIntCopy[0]);// - ccLeft; //m_clipsMask.scannedMask[0][2 * count - 1] + isSet(m_clipsMask.mask[0][2 * count - 1]);
 #else
                m_dthAsyncIntCopy[0];
#endif
            //static uint totalEventCount = 0;
            //totalEventCount += 2 * count;
            //__ct_printf("%d - %f \n", totalEventCount, eventCount / (float)(2 * count));
            //__ct_printf("eventCount=%d %d %d\n", eventCount, m_dthAsyncIntCopy[0], m_dthAsyncIntCopy[1]);
//             if(nodeCount > 15)
//             {
//                 for(int a = 0; a < 1; ++a)
//                 {
//                     CTuint index = m_eventLines.toggleIndex;
//     //                 PrintBuffer(m_eventLines.eventLines[a].indexedEvent[index], eventCount);
//     //                 PrintBuffer(m_eventLines.eventLines[a].primId[index], eventCount);
//     //                 PrintBuffer(m_eventLines.eventLines[a].ranges[index], eventCount);
//     //                 PrintBuffer(m_eventLines.eventLines[a].type[index], eventCount);
//                     PrintBuffer((*m_eventLines.eventLines[a].nodeIndex).Get(index), eventCount, "\n");
//                     __ct_printf("\n\n");
//                 }
//                 printf("hel");
//             }

//             if(eventCount == 0)
//             {
//                  __ct_printf("cuKDTreeScan: FATAL ERROR eventCount %d \n", eventCount);
//                 exit(0);
//             }
            
//             PrintBuffer(m_eventLines.eventLines[0].nodeIndex->Get(m_eventLines.toggleIndex), eventCount);
//             PrintBuffer(m_eventLines.eventLines[0].mask, 2 * count);
//             PrintBuffer(m_clipsMask.scannedMask[0], 2 * count);

            //exit(0);

            m_dthAsyncByteCopy.WaitForStream(m_stream);
            m_dthAsyncByteCopy.StartCopy(m_gotLeaves.GetConstPointer(), 0);
           
            eventBlock = EVENT_GROUP_SIZE;
            eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

            //merge into compact events?
            KERNEL_PROFILE(cudaSetEventsBelongToLeafAndSetNodeIndex(
                m_activeNodesIsLeaf.GetPointer() + nodeCount,
                m_eventIsLeaf.GetPointer(),
                m_nodes_NodeIdToLeafIndex.GetPointer() + g_childNodeOffset,
                eventCount,
                2 * nodeCount,
                m_eventLines.toggleIndex,
                eventGrid, eventBlock, m_pStream), 0cudaSetEventsBelongToLeafAndSetNodeIndex);

            DEVICE_SYNC_CHECK();

            //PROFILE_END;

            //if(!m_dthAsyncByteCopy[0])
            {
                m_interiorContentScanner.Resize(childCount);
                KERNEL_PROFILE(m_interiorContentScanner.ExcScan(m_nodes_ContentCount.Begin(), m_nodes_ContentCount.Begin() + childCount, nutty::PrefixSumOp<CTuint>(), m_pStream), 0m_interiorContentScanner);
                
                DEVICE_SYNC_CHECK();

                CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(
                    m_nodes_ContentStartAdd.GetPointer(), m_interiorContentScanner.GetPrefixSum().GetConstPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
                //nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_interiorContentScanner.GetPrefixSum().Begin(), m_interiorContentScanner.GetPrefixSum().Begin() + childCount);
                DEVICE_SYNC_CHECK();
            }

            m_dthAsyncByteCopy.WaitForCopy();

            DEVICE_SYNC_CHECK();
            KERNEL_PROFILE(leavesRes = MakeLeaves(
                m_activeNodesIsLeaf.Begin(),
                g_childNodeOffset, 
                nodeCount, 
                childCount,
                eventCount, 
                g_currentLeafCount + lastLeaves, 
                g_leafContentOffset, 1, 
                m_dthAsyncByteCopy[0]), MakeLeaves);

            DEVICE_SYNC_CHECK();

            eventCount = 2 * leavesRes.interiorPrimitiveCount;
        }
        else
        {
            //todo
            for(CTuint i = 0; i < nodeCount; ++i)
            {
                m_nodes_IsLeaf.Insert(g_nodeOffset + i, (CTbyte)1);
            }

            __ct_printf("errr not good...\n");
        }

        g_entries += 2 * nodeCount;
        eventSum += eventCount;
        g_interiorNodesCountOnThisLevel = 2 * (nodeCount - lastLeaves) - leavesRes.leafCount;

        g_currentInteriorNodesCount += g_interiorNodesCountOnThisLevel;
        g_nodeOffset = g_childNodeOffset;
        g_childNodeOffset += 2 * (nodeCount);

        //update globals
        g_leafContentOffset += leavesRes.leafPrimitiveCount;
        
        g_currentLeafCount += lastLeaves + leavesRes.leafCount;

        ct_printf(
            "g_nodeOffset=%d g_childNodeOffset=%d g_leafContentOffset=%d g_interiorNodesCountOnThisLevel=%d g_currentInteriorNodesCount=%d g_currentLeafCount=%d\nCreated '%d' Leaves, Interior Nodes '%d'\n", 
              g_nodeOffset, g_childNodeOffset, g_leafContentOffset, g_interiorNodesCountOnThisLevel, g_currentInteriorNodesCount, g_currentLeafCount, lastLeaves + leavesRes.leafCount, g_interiorNodesCountOnThisLevel);

        DEVICE_SYNC_CHECK();

        if(!leavesRes.leafCount)
        {
            cudaMemcpyAsync(m_nodesBBox[0].GetPointer(), m_nodesBBox[1].GetConstPointer(), g_interiorNodesCountOnThisLevel * sizeof(BBox), cudaMemcpyDeviceToDevice, m_pStream);
            //nutty::Copy(m_nodesBBox[0].Begin(), m_nodesBBox[1].Begin(), m_nodesBBox[1].Begin() + g_interiorNodesCountOnThisLevel);
        }

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
                __ct_printf("need memory...\n");
                GrowSplitMemory(2 * eventCount);
            }

            if(m_activeNodes.Size() < g_interiorNodesCountOnThisLevel + 2 * g_interiorNodesCountOnThisLevel)
            {
                __ct_printf("need memory...\n");
                GrowPerLevelNodeMemory(4 * 2 * g_interiorNodesCountOnThisLevel);
            }

            if(m_nodes_IsLeaf.Size() < (g_childNodeOffset + 2 * g_interiorNodesCountOnThisLevel))
            {
                __ct_printf("need memory...\n");
                GrowNodeMemory();
            }
        }
    }

#ifdef PROFILE
    static int frame = 0;
    cudaDeviceSynchronize();
    g_timer.Stop();
    __ct_printf("Total: %f, Section: %f \n", g_timer.GetMillis(), g_time);

    if(frame == PROFILE_FRAMES)
    {
        for(auto it = g_profileTable.begin(); it != g_profileTable.end(); ++it)
        {
            __ct_printf("%s\n", it->first.c_str());
        }

        for(auto it = g_profileTable.begin(); it != g_profileTable.end(); ++it)
        {
            __ct_printf("%f\n", it->second / (float)PROFILE_FRAMES);
        }
        g_profileTable.clear();
        exit(0);
    }

    frame++;
#endif

    m_interiorNodesCount = g_currentInteriorNodesCount;
    m_leafNodesCount = g_currentLeafCount;
    //CTuint allNodeCount = m_interiorNodesCount + m_leafNodesCount;
#ifdef _DEBUG
   // CUDA_RT_SAFE_CALLING_NO_SYNC(cudaStreamSynchronize(m_pStream));
    __ct_printf("%d\n", eventSum);
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
#endif

    DEVICE_SYNC_CHECK();

#ifdef _DEBUG
    ValidateTree();
#endif

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
#else

CT_RESULT cuKDTreeScan::Update(void)
{
    return CT_INVALID_OPERATION;
}
#endif