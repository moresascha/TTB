
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif
#include <cutil_math.h>
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

template<>
struct ShrdMemory<CTuint3>
{
    __device__ CTuint3* Ptr(void) 
    { 
        extern __device__ __shared__ CTuint3 s_b4[];
        return s_b4;
    }
};

#define PROFILE
#ifdef PROFILE
#define PROFILE_START chimera::util::HTimer timer; cudaDeviceSynchronize(); timer.Start()
#define PROFILE_END cudaDeviceSynchronize(); timer.Stop(); g_time += timer.GetMillis()
#else
#define PROFILE_START
#define PROFILE_END
#endif

#define PRINT_OUT
#ifndef _DEBUG
#undef PRINT_OUT
#endif

#ifndef PRINT_OUT
#undef PRINT_BUFFER(_name)
#undef PRINT_BUFFER_N(_name, _tmp)
#undef PRINT_RAW_BUFFER
#undef PRINT_RAW_BUFFER_N
#undef ct_printf

#define PRINT_BUFFER(_name)
#define PRINT_BUFFER_N(_name, _tmp)
#define PRINT_RAW_BUFFER(_name)
#define PRINT_RAW_BUFFER_N(_name, _N)
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

    __device__ __host__ T GetNeutral(void)
    {
        return 1;
    }
};

struct ScanByte3
{
    __device__ CTuint3 operator()(CTbyte3 elem)
    {
        CTuint3 v;
        v.x = elem.x ^ 1;
        v.y = elem.y ^ 1;
        v.z = elem.z ^ 1;
        return v;
    }

    __device__ __host__ CTbyte3 GetNeutral(void)
    {
        CTbyte3 v;
        v.x = 1; v.y = 1; v.z = 1;
        return v;
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

void cuKDTreeScan::InitBuffer(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;

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

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_nodes, &m_nodes, sizeof(Node)));
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

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_nodes, &m_nodes, sizeof(Node)));
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

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_splits, &m_splits, sizeof(Split)));
    
    SplitConst splitsConst;
    splitsConst.above = m_splits_Above.GetDevicePtr()();
    splitsConst.below = m_splits_Below.GetDevicePtr()();
    splitsConst.axis = m_splits_Axis.GetDevicePtr()();
    splitsConst.indexedSplit = m_splits_IndexedSplit.GetDevicePtr()();
    splitsConst.v = m_splits_Plane.GetDevicePtr()();

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_splitsConst, &splitsConst, sizeof(SplitConst)));
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
    const CTuint* nodesContentCount)
{
    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
    //cuEventLineTriple tripleLine(m_events3, 0);
    CTuint start = 0;

    m_pool.Reset();
    m_typeScanner.Resize(eventCount);
    m_types3.Resize(eventCount);
    
    //make3AxisType<<<eventGrid, eventBlock>>>(m_types3.Begin()(), tripleLine, eventCount);
    //m_typeScanner.ExcScan(m_types3.Begin(), m_types3.Begin() + eventCount, ScanByte3());
    //makeSeperateScans<<<eventGrid, eventBlock>>>(tripleLine, m_typeScanner.GetPrefixSum().Begin()(), eventCount);
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

    static const CTuint blockS = 256;
    static const CTuint grid = 3;
    //static EventStartScanOp<CTbyte> op0;
    CTbyte add = m_eventLines.toggleIndex;

    ConstTuple<3, CTbyte> ptr;
    ptr.ts[0] = m_eventLines.eventLines[0].type[add].GetConstPointer();
    ptr.ts[1] = m_eventLines.eventLines[1].type[add].GetConstPointer();
    ptr.ts[2] = m_eventLines.eventLines[2].type[add].GetConstPointer();
    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = m_eventLines.eventLines[0].typeStartScanned.GetPointer();
    ptr1.ts[1] = m_eventLines.eventLines[1].typeStartScanned.GetPointer();
    ptr1.ts[2] = m_eventLines.eventLines[2].typeStartScanned.GetPointer();
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = m_eventLines.eventLines[0].scannedEventTypeEndMaskSums.GetPointer();
    sums.ts[1] = m_eventLines.eventLines[1].scannedEventTypeEndMaskSums.GetPointer();
    sums.ts[2] = m_eventLines.eventLines[2].scannedEventTypeEndMaskSums.GetPointer();

    ConstTuple<3, CTuint> constSums;
    constSums.ts[0] = m_eventLines.eventLines[0].scannedEventTypeEndMaskSums.GetConstPointer();
    constSums.ts[1] = m_eventLines.eventLines[1].scannedEventTypeEndMaskSums.GetConstPointer();
    constSums.ts[2] = m_eventLines.eventLines[2].scannedEventTypeEndMaskSums.GetConstPointer();

 #if 1
    binaryTripleGroupScan<blockS><<<eventGrid, eventBlock>>>(
        ptr, ptr1, sums,
        eventCount);
#else

    for(int i = 0; i < 3; ++i)
    {
        shfl_scan_perblock<blockS><<<eventGrid, eventBlock>>>(
            m_eventLines.eventLines[i].type[add].GetConstPointer(),
            m_eventLines.eventLines[i].typeStartScanned.GetPointer(),
            m_eventLines.eventLines[i].scannedEventTypeEndMaskSums.GetPointer(), 
            op0,
            eventCount);
    }
#endif
   
    CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, blockS);
    //__ct_printf("%d\n", sumsCount);
    nutty::PrefixSumOp<CTuint> op;

#if 1
    completeScan2<256, 3><<<3, 256>>>(constSums, sums, op, sumsCount);
#else
    CTuint shrdStepElemperThread = nutty::cuda::GetCudaGrid(sumsCount, 256U);

    switch(shrdStepElemperThread)
    {
    case  1: completeScan<256, 3, 1><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  2: completeScan<256, 3, 2><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  3: completeScan<256, 3, 3><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  4: completeScan<256, 3, 4><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  5: completeScan<256, 3, 5><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  6: completeScan<256, 3, 6><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  7: completeScan<256, 3, 7><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  8: completeScan<256, 3, 8><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case  9: completeScan<256, 3, 9><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case 10: completeScan<256, 3, 10><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case 11: completeScan<256, 3, 11><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    case 12: completeScan<256, 3, 12><<<3, 256>>>(constSums, sums, op, sumsCount); break;
    default:   __ct_printf("error\n"); exit(0); break;
    };
#endif

   // PRINT_BUFFER_N(m_eventLines.eventLines[0].scannedEventTypeEndMaskSums, sumsCount);
    if(eventGrid-1)
    {
        spreadScannedSums<<<eventGrid-1, eventBlock>>>(ptr1, sums, eventCount);
    }
    
//     if(cudaSuccess != cudaDeviceSynchronize())
//     {
//         __ct_printf("error\n");
//        //exit(0);
//     }

    
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
    CTuint N = nutty::cuda::GetCudaGrid(eventCount, elemsPerThread);
    CTuint sahBlock = nutty::cuda::GetCudaBlock(N, 256U);
    CTuint sahGrid = nutty::cuda::GetCudaGrid(N, sahBlock);

    computeSAHSplits3<1, elemsPerThread><<<sahGrid, sahBlock>>>(
        nodesContentCount,
        m_nodes_ContentStartAdd.Begin()(),
        m_nodesBBox[0].Begin()(),
        eventCount,
        m_eventLines.toggleIndex);

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

    start = 0;
    m_pool.Reset();

#if defined DYNAMIC_PARALLELISM
    if(true) //5
    {
//         CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 32U);
//         CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);
//         dpReduceSAHSplits<<<nodeGrid, nodeBlock>>>(m_splits_IndexedSplit.GetPointer(), nodeCount);

        if(nodeCount == 1)
        {
            IndexedSAHSplit neutralSplit;
            neutralSplit.index = 0;
            neutralSplit.sah = FLT_MAX;
            nutty::Reduce(m_splits_IndexedSplit.Begin(), m_splits_IndexedSplit.Begin() + eventCount, ReduceIndexedSplit(), neutralSplit);
        }
//         else if(nodeCount == 2)
//         {
//             CTuint nodeBlock = nodeCount;//nutty::cuda::GetCudaBlock(nodeCount, 32U);
//             CTuint nodeGrid = 1;
//             dpReduceSAHSplits<<<nodeGrid, nodeBlock>>>(m_splits_IndexedSplit.GetPointer(), nodeCount);
//         }
        else
        {
            const CTuint blockSize = 512U;
            CTuint N = nodeCount * blockSize;
            CTuint reduceBlock = nutty::cuda::GetCudaBlock(N, blockSize);
            CTuint reduceGrid = nutty::cuda::GetCudaGrid(N, reduceBlock);
            segReduce<blockSize><<<reduceGrid, reduceBlock>>>(m_splits_IndexedSplit.GetPointer(), N);
        }

#ifdef PRINT_OUT
#if 0
        m_dthAsyncNodesContent.WaitForCopy();
        for(int i = 0; i < nodeCount; ++i)
        {
            CTuint cc = m_dthAsyncNodesContent[i];
            CTuint length = 2 * cc;

            IndexedSAHSplit s = *(m_splits_IndexedSplit.Begin() + start);
            std::stringstream ss;
            ss << m_nodesBBox[0][i];
            __ct_printf("%s ", ss.str().c_str());
            __ct_printf("id=%d, memoryadd=%d ", s.index, start);
            CTreal plane = m_splits_Plane[s.index];
            CTbyte axis = m_splits_Axis[s.index];
            CTuint below = m_splits_Below[s.index];
            CTuint above = m_splits_Above[s.index];
            __ct_printf("axis=%d plane=%f sah=%f below=%d above=%d\n", (CTuint)axis, plane, s.sah, below, above);

            start += length;
        }
#endif
#endif
    }
    else
#endif
    {
        m_dthAsyncNodesContent.WaitForCopy();
        for(int i = 0; i < nodeCount; ++i)
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

                        BBox bbox;// = m_nodesBBox[0][ m_events3[0].nodeIndex[m_events3[0].toggleIndex][i] ];
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
}

CTuint cuKDTreeScan::CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint nodeRange)
{
    m_leafCountScanner.Resize(nodeRange);

    m_leafCountScanner.ExcScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, TypeOp<CTbyte>());

    m_dthAsyncIntCopy.StartCopy(m_leafCountScanner.GetPrefixSum().GetConstPointer() + nodeRange - 1, 0);
    m_dthAsyncByteCopy.StartCopy(isLeafBegin() + nodeOffset + nodeRange - 1, 0);
    //CTuint tt = m_leafCountScanner.GetPrefixSum()[nodeRange-1] + (*(isLeafBegin + nodeOffset + nodeRange - 1) == 1);

    /*if(!leafCount)
    {
        return 0;
    } */

    CTuint block = nutty::cuda::GetCudaBlock(nodeRange, 256U);
    CTuint grid = nutty::cuda::GetCudaGrid(nodeRange, block);

    if(m_interiorCountScanned.Size() <= nodeRange)
    {
        m_interiorCountScanned.Resize(nodeRange);
        m_maskedInteriorContent.Resize(nodeRange);
        m_interiorContentScanner.Resize(nodeRange);
        m_leafContentScanned.Resize(nodeRange);
    }

    createInteriorContentCountMasks<<<grid, block>>>(
        isLeafBegin() + nodeOffset,
        m_nodes_ContentCount.Begin()(), 
        m_maskedInteriorContent.Begin()(), nodeRange);

    m_interiorContentScanner.ExcScan(m_maskedInteriorContent.Begin(), m_maskedInteriorContent.Begin() + nodeRange, nutty::PrefixSumOp<CTuint>());
    
    makeOthers<<<grid, block>>>(

        m_nodes_ContentStartAdd.Begin()(), 
        m_interiorContentScanner.GetPrefixSum().Begin()(), 
        m_leafContentScanned.Begin()(), 
        
        m_leafCountScanner.GetPrefixSum().Begin()(), 
        m_interiorCountScanned.Begin()(), 
        
        nodeRange);

    m_dthAsyncIntCopy.WaitForCopy();
    m_dthAsyncByteCopy.WaitForCopy();
    CTuint leafCount = m_dthAsyncIntCopy[0] + (m_dthAsyncByteCopy[0] == 1);
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
    CTuint initNodeToLeafIndex,
    CTbyte gotLeaves)
{

    CTuint leafCount = 0;
    if(gotLeaves)
    {
         leafCount = CheckRangeForLeavesAndPrepareBuffer(isLeafBegin, nodeOffset, nodeCount);
    }
     
    if(!leafCount)
    {
        MakeLeavesResult result;
        result.leafCount = 0;
        result.interiorPrimitiveCount = eventCount/2;
        result.leafPrimitiveCount = 0;
        return result;
    }

    m_dthAsyncIntCopy.StartCopy(m_leafContentScanned.Begin()() + nodeCount - 1, 0);
    m_dthAsyncIntCopy.StartCopy(m_nodes_ContentCount.Begin()() + nodeCount - 1, 1);
    m_dthAsyncByteCopy.StartCopy(m_activeNodesIsLeaf.Begin()() + nodeCount + nodeOffset - 1, 0);
 
    m_leafNodesContentStart.Resize(currentLeafCount + leafCount);
    m_leafNodesContentCount.Resize(currentLeafCount + leafCount);

    CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
    CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);
     
    m_eventIsLeafScanner.Resize(eventCount);
    m_eventIsLeafScanner.ExcScan(m_eventIsLeaf.Begin(), m_eventIsLeaf.Begin() + eventCount, TypeOp<CTuint>());
    
    if(m_leafNodesContent.Size() < leafContentOffset + eventCount/2)
    {
        m_leafNodesContent.Resize(leafContentOffset + eventCount/2);
    }

     compactMakeLeavesData<<<eventGrid, eventBlock>>>(
         isLeafBegin() + nodeOffset,
         m_interiorCountScanned.GetPointer(),
         m_leafContentScanned.GetPointer(),
         m_eventIsLeafScanner.GetPrefixSum().GetConstPointer(),
            
         m_nodes_ContentCount.GetPointer(),
         m_eventIsLeaf.GetPointer(),

         m_leafCountScanner.GetPrefixSum().GetConstPointer(), 
         m_interiorCountScanned.GetConstPointer(),

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
         eventCount);
    
    DEVICE_SYNC_CHECK();

    m_eventLines.Toggle();

    CTuint copyDistance = nodeCount - leafCount;

    if(copyDistance)
    {
        cudaMemcpyAsync(m_nodes_ContentCount.GetPointer(), m_newNodesContentCount.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(m_nodes_ContentStartAdd.GetPointer(), m_newNodesContentStartAdd.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(m_activeNodes.GetPointer(), m_newActiveNodes.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);
    }

    m_dthAsyncIntCopy.WaitForCopy();
    m_dthAsyncByteCopy.WaitForCopy();

    CTuint leafPrimCount = m_dthAsyncIntCopy[0] + m_dthAsyncByteCopy[0] * m_dthAsyncIntCopy[1]; 

    CTuint interiorPrimCount = eventCount/2 - leafPrimCount;
    interiorPrimCount = interiorPrimCount > eventCount/2 ? 0 : interiorPrimCount;

    MakeLeavesResult result;
    result.leafCount = leafCount;
    result.interiorPrimitiveCount = interiorPrimCount;
    result.leafPrimitiveCount = leafPrimCount;

    return result;
}

void EventLine::ScanEvents(CTuint length)
{
    eventScanner.ExcScan(mask.Begin(), mask.Begin() + length, nutty::PrefixSumOp<CTbyte>());
}

struct ClipMaskPrefixSumOP
{
    __device__ CTbyte operator()(CTbyte elem)
    {
        return isSet(elem) ? 1 : 0;
    }

    __device__ __host__ CTbyte GetNeutral(void)
    {
        return 0;
    }
};

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
    for(CTbyte i = 0; i < 3; ++i)
    {
        maskScanner[i].ExcScan(mask[i].Begin(), mask[i].Begin() + length, ClipMaskPrefixSumOP());
    }

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

void EventLines::BindToConstantMemory(void)
{
    cuEventLineTriple src;//(eventLines, 0);
    src.lines[0] = eventLines[0].GetPtr(0);
    src.lines[1] = eventLines[1].GetPtr(0);
    src.lines[2] = eventLines[2].GetPtr(0);


    cuEventLineTriple dst;//(eventLines, 1);
    dst.lines[0] = eventLines[0].GetPtr(1);
    dst.lines[1] = eventLines[1].GetPtr(1);
    dst.lines[2] = eventLines[2].GetPtr(1);

    cudaMemcpyToSymbol(g_eventTriples, &src, sizeof(cuEventLineTriple));
    cudaMemcpyToSymbol(g_eventTriples, &dst, sizeof(cuEventLineTriple), sizeof(cuEventLineTriple));
}

// void EventLines::BindToggleIndexToConstantMemory(void)
// {
//     CTbyte dst = ((toggleIndex+1)%2);
//     cudaMemcpyToSymbol(g_eventSrcIndex, &toggleIndex, sizeof(CTbyte));
//     cudaMemcpyToSymbol(g_eventDstIndex, &dst, sizeof(CTbyte));
// }

CT_RESULT cuKDTreeScan::Update(void)
{
    if(!m_initialized)
    {
        InitBuffer();
        m_initialized = true;
    }

    ClearBuffer();
    
    CTuint primitiveCount = m_currentTransformedVertices.Size() / 3;

    uint _block = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    uint _grid = nutty::cuda::GetCudaGrid(primitiveCount, _block);

    cudaCreateTriangleAABBs(m_currentTransformedVertices.GetPointer(), m_primAABBs.GetPointer(), primitiveCount);

    DEVICE_SYNC_CHECK();

    static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
    static float3 min3f = -max3f;

    BBox bboxN;
    bboxN.m_min = max3f; 
    bboxN.m_max = min3f;
    m_sceneBBox.Resize(m_primAABBs.Size()/2);
    nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), ReduceBBox(), bboxN);

    DEVICE_SYNC_CHECK(); 
    
    CTuint elementBlock = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);
    
    m_eventLines.Resize(2 * primitiveCount);

#ifdef PROFILE
        chimera::util::HTimer g_timer;
        cudaDeviceSynchronize();
        g_timer.Start();
        g_time = 0;
#endif

    m_eventLines.toggleIndex = 0;

    createEventsAndInit3<1, 0><<<elementGrid, elementBlock>>>(
        m_primAABBs.GetConstPointer(), 
        m_sceneBBox.GetConstPointer(),

        m_activeNodes.GetPointer(),
        m_nodes_NodeIdToLeafIndex.GetPointer(),
        m_nodes_IsLeaf.GetPointer(),
        m_nodes_ContentCount.GetPointer(),
        m_nodesBBox[0].GetPointer(),

        primitiveCount);

    DEVICE_SYNC_CHECK();
    
    for(CTbyte i = 0; i < 3; ++i)
    {
        nutty::Sort(
            nutty::DevicePtr_Cast<IndexedEvent>(m_eventLines.eventLines[i].GetPtr(0).indexedEvent), 
            nutty::DevicePtr_Cast<IndexedEvent>(m_eventLines.eventLines[i].GetPtr(0).indexedEvent + 2 * primitiveCount), 
            EventSort());
    }

    DEVICE_SYNC_CHECK();

    reorderEvent3<<<2 * elementGrid, elementBlock>>>(2 * primitiveCount);

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
    
    CTuint eventCount = 2 * primitiveCount;

    m_eventLines.Toggle();
    
    CTuint maxDepth = 0;
    for(CTbyte d = 0; d <= m_depth; ++d)
    {
        
        ct_printf("\nNew Level=%d (%d)\n", d, m_depth);
//         for(int i = 0; i < 3 && d>=6; ++i)
//         {
//           //  PrintEventLine(m_events3[i], eventCount);
//         }

        CTuint nodeCount = g_interiorNodesCountOnThisLevel;
        CTuint nodeBlock = nutty::cuda::GetCudaBlock(nodeCount, 256U);
        CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        CTuint eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
        CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

        DEVICE_SYNC_CHECK();

#if defined _DEBUG
#if defined DYNAMIC_PARALLELISM
        //if(nodeCount < 0) //6
        {
#endif
            //m_hNodesContentCount.Resize(nodeCount);
            m_dthAsyncNodesContent.Resize(nodeCount);
            m_dthAsyncNodesContent.StartCopy(m_nodes_ContentCount.GetConstPointer(), 0, nodeCount);
            //nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);

#if defined DYNAMIC_PARALLELISM
        }
#endif
#endif

#ifdef PRINT_OUT
        m_hNodesContentCount.Resize(nodeCount);
        nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);
#endif

        PRINT_BUFFER_N(m_nodes_ContentCount, nodeCount);

        //m_pool.ClearEvents();

        ComputeSAH_Splits(
            nodeCount, 
            eventCount,
            m_nodes_ContentCount.Begin()());
        
        makeLeafIfBadSplitOrLessThanMaxElements<<<nodeGrid, nodeBlock>>>(
            m_nodes,
            m_nodes_IsLeaf.GetPointer() + g_nodeOffset,
            m_activeNodes.GetPointer(),
            m_activeNodesIsLeaf.GetPointer(), 
            m_splits,
            d == m_depth-1,
            nodeCount);
        
        m_newNodesContentCount.Resize(m_nodes_ContentCount.Size());
        m_newNodesContentStartAdd.Resize(m_nodes_ContentCount.Size());

        m_lastNodeContentStartAdd.Resize(m_newNodesContentStartAdd.Size());

        cudaMemcpyAsync(m_lastNodeContentStartAdd.GetPointer(), m_nodes_ContentStartAdd.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);

        MakeLeavesResult leavesRes;// = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_nodeOffset, 0, nodeCount, eventCount, g_currentLeafCount, g_leafContentOffset, 0);
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
PROFILE_START;
            CTuint block = nutty::cuda::GetCudaBlock(count, 256U);
            CTuint grid = nutty::cuda::GetCudaGrid(count, block);    

            m_eventLines.Resize(2 * count);

#define COMPACT_EVENT_V2
            
            CTuint _block = nutty::cuda::GetCudaBlock(2 * count, 256U);
            CTuint _grid = nutty::cuda::GetCudaGrid(2 * count, block);

#ifdef COMPACT_EVENT_V2
            m_clipsMask.Resize(2 * count);

            clearMasks3<<<_grid, _block>>>(m_clipsMask.mask3.GetPointer(), 2 * count);

            cuClipMaskArray mm;
            m_clipsMask.GetPtr(mm);
            
            createClipMask<<<grid, block>>>(
                mm, 
                (CTbyte*)m_clipsMask.mask3.GetPointer(),
                m_nodes_ContentStartAdd.GetPointer(), 
                m_nodes_ContentCount.GetPointer(),
                count,
                m_eventLines.toggleIndex);
            
            //m_clipsMask.ScanMasks(2 * count);
            m_clipsMask.mask3Scanner.ExcScan(m_clipsMask.mask3.Begin(), m_clipsMask.mask3.Begin() + 2 * count, ClipMaskPrefixSum3OP());

            //PRINT_BUFFER_N(m_clipsMask.mask3, 2 * count);
            //PRINT_BUFFER_N(m_clipsMask.mask3Scanner.GetPrefixSum(), 2 * count);

//            Sums bb;
//             bb.prefixSum[0] = m_clipsMask.maskScanner[0].GetPrefixSum().GetConstPointer();
//             bb.prefixSum[1] = m_clipsMask.maskScanner[1].GetPrefixSum().GetConstPointer();
//             bb.prefixSum[2] = m_clipsMask.maskScanner[2].GetPrefixSum().GetConstPointer();

            compactEventLineV2<<<_grid, _block>>>(
                mm,
                (CTbyte*)m_clipsMask.mask3.GetConstPointer(),
                //bb, 
                (CTuint*)m_clipsMask.mask3Scanner.GetPrefixSum().GetConstPointer(),
                2 * count,
                m_eventLines.toggleIndex);

#elif defined COMPACT_EVENT_V1
            
            clearMasks<<<_grid, _block>>>(
            m_eventLines.eventLines[0].mask.Begin()(),
            m_eventLines.eventLines[1].mask.Begin()(), 
            m_eventLines.eventLines[2].mask.Begin()(),
            2 * count);

            clipEvents3<<<grid, block>>>(
                m_nodes_ContentStartAdd.Begin()(), 
                m_nodes_ContentCount.Begin()(),
                count,
                m_eventLines.toggleIndex);
               
//             for(CTbyte i = 0; i < 3; ++i)
//             {
//                 clipEvents<<<grid, block>>>(
//                     m_events3[i].GetDst(), 
//                     m_events3[i].GetSrc(), 
//                     m_events3[i].mask.Begin()(), 
//                     m_nodes_ContentStartAdd.Begin()(), 
//                     m_nodes_ContentCount.Begin()(),
//                     m_splits, 
//                     i, 
//                     count);
//             }
            
             
            DEVICE_SYNC_CHECK();
            
            m_eventLines.Toggle();
#endif

#ifdef COMPACT_EVENT_V1
            for(CTbyte i = 0; i < 3; ++i)
            {
                m_eventLines.eventLines[i].ScanEvents(2 * count);
                DEVICE_SYNC_CHECK();
            }

            Sums bb;
            bb.prefixSum[0] = m_eventLines.eventLines[0].eventScanner.GetPrefixSum().Begin()();
            bb.prefixSum[1] = m_eventLines.eventLines[1].eventScanner.GetPrefixSum().Begin()();
            bb.prefixSum[2] = m_eventLines.eventLines[2].eventScanner.GetPrefixSum().Begin()();
 
            compactEventLine3<<<_grid, _block>>>(
                bb, 2 * count,
                m_eventLines.toggleIndex);
#endif
            
            m_eventLines.Toggle();

            DEVICE_SYNC_CHECK();

#ifdef COMPACT_EVENT_V1

            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_eventLines.eventLines[0].eventScanner.GetPrefixSum().GetConstPointer() + count - 1), 0);
            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_eventLines.eventLines[0].eventScanner.GetPrefixSum().GetConstPointer() + 2 * count - 1), 1);

            m_dthAsyncByteCopy.StartCopy((CTbyte*)(m_eventLines.eventLines[0].mask.GetPointer() + count - 1), 0);
            m_dthAsyncByteCopy.StartCopy((CTbyte*)(m_eventLines.eventLines[0].mask.GetPointer() + 2 * count - 1), 1);
#else
//             m_dthAsyncIntCopy.StartCopy((*m_clipsMask.maskScanner[0].GetPrefixSum().GetRawPointer()) + count - 1, 0);
//             m_dthAsyncIntCopy.StartCopy((*m_clipsMask.maskScanner[0].GetPrefixSum().GetRawPointer()) + 2 * count - 1, 1);
// 
//             m_dthAsyncByteCopy.StartCopy((*m_clipsMask.mask[0].GetRawPointer()) + count - 1, 0);
//             m_dthAsyncByteCopy.StartCopy((*m_clipsMask.mask[0].GetRawPointer()) + 2 * count - 1, 1);

            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.mask3Scanner.GetPrefixSum().GetConstPointer() + count - 1), 0);
            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.mask3Scanner.GetPrefixSum().GetConstPointer() + 2 * count - 1), 1);

            m_dthAsyncByteCopy.StartCopy((CTbyte*)(m_clipsMask.mask3.GetPointer() + count - 1), 0);
            m_dthAsyncByteCopy.StartCopy((CTbyte*)(m_clipsMask.mask3.GetPointer() + 2 * count - 1), 1);
#endif
            g_leafContentOffset += leavesRes.leafPrimitiveCount;
PROFILE_END;
            DEVICE_SYNC_CHECK();
            
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

            initInteriorNodes<<<nodeGrid, nodeBlock>>>(
                m_activeNodes.Begin()(),
                m_activeNodesThisLevel.Begin()(),

                m_nodesBBox[0].GetPointer(), 
                m_nodesBBox[1].GetPointer(), 

                m_newNodesContentCount.GetPointer(),
                m_newNodesContentStartAdd.GetPointer(),
                m_newActiveNodes.GetPointer(),
                m_activeNodesIsLeaf.GetPointer() + nodeCount,

                g_childNodeOffset,
                g_nodeOffset,
                thisLevelNodesLeft,
                m_lastNodeContentStartAdd.GetPointer(),
                m_gotLeaves.GetPointer(),
                m_depth == d+1);

            cudaMemcpyAsync(m_activeNodes.GetPointer(), m_newActiveNodes.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(m_nodes_ContentCount.GetPointer(), m_newNodesContentCount.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(m_nodes_ContentStartAdd.GetPointer(), m_newNodesContentStartAdd.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);

            m_dthAsyncIntCopy.WaitForCopy();
            m_dthAsyncByteCopy.WaitForCopy();
            cudaDeviceSynchronize();
 
            CTuint ccLeft = m_dthAsyncIntCopy[0] + isSet(m_dthAsyncByteCopy[0]);
            CTuint ccRight = m_dthAsyncIntCopy[1] + isSet(m_dthAsyncByteCopy[1]) - ccLeft;

            m_dthAsyncByteCopy.StartCopy(m_gotLeaves.GetConstPointer(), 0);
            
            eventCount = ccRight + ccLeft;
            if(eventCount == 0)
            {
                __ct_printf("error: eventCount == 0\n");
                exit(0);
            }
            eventBlock = nutty::cuda::GetCudaBlock(eventCount, 256U);
            eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
            
            setEventsBelongToLeafAndSetNodeIndex<<<eventGrid, eventBlock>>>(
                m_activeNodesIsLeaf.GetPointer() + nodeCount,
                m_eventIsLeaf.GetPointer(),
                m_nodes_NodeIdToLeafIndex.GetPointer() + g_childNodeOffset,
                eventCount,
                2 * nodeCount,
                m_eventLines.toggleIndex);

            //PROFILE_END;

            //if(!m_dthAsyncByteCopy[0])
            {
                m_interiorContentScanner.Resize(childCount);
                m_interiorContentScanner.ExcScan(m_nodes_ContentCount.Begin(), m_nodes_ContentCount.Begin() + childCount, nutty::PrefixSumOp<CTuint>());
                //cudaMemcpyAsync(
                
                cudaMemcpyAsync(m_nodes_ContentStartAdd.GetPointer(), m_interiorContentScanner.GetPrefixSum().GetConstPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice);
                //nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_interiorContentScanner.GetPrefixSum().Begin(), m_interiorContentScanner.GetPrefixSum().Begin() + childCount);
            }

            m_dthAsyncByteCopy.WaitForCopy();

            leavesRes = MakeLeaves(m_activeNodesIsLeaf.Begin(), g_childNodeOffset, nodeCount, childCount, eventCount, g_currentLeafCount + lastLeaves, g_leafContentOffset, 1, m_dthAsyncByteCopy[0]);

            eventCount = 2 * leavesRes.interiorPrimitiveCount;
        }
        else
        {
            //todo
            for(int i = 0; i < nodeCount; ++i)
            {
                m_nodes_IsLeaf.Insert(g_nodeOffset + i, (CTbyte)1);
            }
            __ct_printf("errr not good...\n");
        }

        g_entries += 2 * nodeCount;
        g_lastChildCount = 2 * nodeCount;
        g_nodeOffset2 = g_nodeOffset;
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
            nutty::Copy(m_nodesBBox[0].Begin(), m_nodesBBox[1].Begin(), m_nodesBBox[1].Begin() + g_interiorNodesCountOnThisLevel);
        }

        PRINT_BUFFER(m_nodesBBox[0]);

//         PRINT_BUFFER_N(m_primNodeIndex, primitiveCount);
//         PRINT_BUFFER_N(m_nodes_LeftChild, g_entries);
//         PRINT_BUFFER_N(m_nodes_RightChild, g_entries);
//         PRINT_BUFFER_N(m_nodes_IsLeaf, g_entries);
//         PRINT_BUFFER_N(m_nodes_NodeIdToLeafIndex, g_entries);

        maxDepth = d;
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
    __ct_printf("Total: %f, Section: %f (maxdepth=%d)\n", g_timer.GetMillis(), g_time, maxDepth);
#endif

    m_interiorNodesCount = g_currentInteriorNodesCount;
    m_leafNodesCount = g_currentLeafCount;
    CTuint allNodeCount = m_interiorNodesCount + m_leafNodesCount;

#ifdef _DEBUG
    ValidateTree();
#endif

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