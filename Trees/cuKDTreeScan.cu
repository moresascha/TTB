
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#undef NUTTY_DEBUG

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
#include <fstream>

#define NODES_GROUP_SIZE 256U
#define EVENT_GROUP_SIZE 256U

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

template<>
struct ShrdMemory<CTuint3>
{
    __device__ CTuint3* Ptr(void) 
    { 
        extern __device__ __shared__ CTuint3 s_b4[];
        return s_b4;
    }
};

#undef PROFILE
#ifdef PROFILE
#define PROFILE_START chimera::util::HTimer timer; cudaDeviceSynchronize(); timer.Start()
#define PROFILE_END cudaDeviceSynchronize(); timer.Stop(); g_time += timer.GetMillis()
#else
#define PROFILE_START
#define PROFILE_END
#endif

#undef PRINT_OUT
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
    __device__ CTuint operator()(T elem)
    {
        return (elem < 2) * elem;
    }

    __device__ __host__ CTuint GetNeutral(void)
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

template <typename Operator, typename T>
void ScanBinaryTriples(ConstTuple<3, T>& src, Tuple<3, CTuint>& scanned, Tuple<3, CTuint>& sums, CTuint N, Operator op, cudaStream_t pStream)
{
    static const CTuint block = 256;

    ConstTuple<3, CTuint> constSums;
    constSums.ts[0] = sums.ts[0];
    constSums.ts[1] = sums.ts[1];
    constSums.ts[2] = sums.ts[2];

    CTuint grid = nutty::cuda::GetCudaGrid(N, block);

    binaryTripleGroupScan<block><<<grid, block, 0, pStream>>>(
        src, scanned, sums, op,
        N);
    
    DEVICE_SYNC_CHECK();

    CTuint sumsCount = nutty::cuda::GetCudaGrid(N, block);

    if(sumsCount > 1)
    {
#if 1
        nutty::PrefixSumOp<CTuint> _op;
        completeScan2<256, 3><<<3, 256, 0, pStream>>>(constSums, sums, _op, sumsCount);

        DEVICE_SYNC_CHECK();
#else
        CTuint shrdStepElemperThread = nutty::cuda::GetCudaGrid(sumsCount, 256U);

        switch(shrdStepElemperThread)
        {
        case  1: completeScan<256, 3, 1><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  2: completeScan<256, 3, 2><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  3: completeScan<256, 3, 3><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  4: completeScan<256, 3, 4><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  5: completeScan<256, 3, 5><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  6: completeScan<256, 3, 6><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  7: completeScan<256, 3, 7><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  8: completeScan<256, 3, 8><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case  9: completeScan<256, 3, 9><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 10: completeScan<256, 3, 10><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 11: completeScan<256, 3, 11><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        case 12: completeScan<256, 3, 12><<<3, 256, 0, pStream>>>(constSums, sums, op, sumsCount); break;
        default:   __ct_printf("error\n"); exit(0); break;
        };
#endif

         spreadScannedSums<<<grid-1, block, 0, pStream>>>(scanned, sums, N);
         DEVICE_SYNC_CHECK();
    }
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

    nutty::PrefixSumOp<CTbyte> op;
    ScanBinaryTriples(ptr, ptr1, sums, eventCount, op, m_pStream); 
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
    m_typeScanner.Resize(eventCount);
    m_types3.Resize(eventCount);
    
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

    computeSAHSplits3<1, elemsPerThread><<<sahGrid, sahBlock, 0, m_pStream>>>(
        nodesContentCount,
        m_nodes_ContentStartAdd.Begin()(),
        m_nodesBBox[0].Begin()(),
        eventCount,
        m_eventLines.toggleIndex);

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

    if(true) //5
    {
        if(nodeCount == 1)
        {
            IndexedSAHSplit neutralSplit;
            neutralSplit.index = 0;
            neutralSplit.sah = FLT_MAX;
            nutty::Reduce(m_splits_IndexedSplit.Begin(), m_splits_IndexedSplit.Begin() + eventCount, ReduceIndexedSplit(), neutralSplit, m_pStream);
            DEVICE_SYNC_CHECK();
        }
#if defined DYNAMIC_PARALLELISM
//         else if(nodeCount == 2)
//         {
//             CTuint nodeBlock = nodeCount;//nutty::cuda::GetCudaBlock(nodeCount, 32U);
//             CTuint nodeGrid = 1;
//             dpReduceSAHSplits<<<nodeGrid, nodeBlock>>>(m_splits_IndexedSplit.GetPointer(), nodeCount);
//         }
#endif
        else
        {
            const CTuint blockSize = 512U;
            CTuint N = nodeCount * blockSize;
            CTuint reduceGrid = nutty::cuda::GetCudaGrid(N, blockSize);
            //cudaErrorBuffer errorBuffer;

            segReduce<blockSize><<<reduceGrid, blockSize, 0, m_pStream>>>(m_splits_IndexedSplit.GetPointer(), N, eventCount);

//             if(errorBuffer.check())
//             {
//                 PrintBuffer(m_nodes_ContentCount, nodeCount);
//                 PrintBuffer(m_nodes_ContentStartAdd, nodeCount);
//                 __debugbreak();
//             }
            DEVICE_SYNC_CHECK();
        }

#if 0
        m_hNodesContentCount.Resize(nodeCount);
        nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);
        for(int i = 0; i < nodeCount; ++i)
        {
            CTuint cc = m_hNodesContentCount[i];
            CTuint length = 2 * cc;

            IndexedSAHSplit s = *(m_splits_IndexedSplit.Begin() + start);

            if(IS_INVALD_SAH(s.sah))
            {
                std::stringstream ss;
                ss << m_nodesBBox[0][i];
                __ct_printf("%s ", ss.str().c_str());
                __ct_printf("id=%d, memoryadd=%d ", s.index, start);
                CTreal plane = m_splits_Plane[s.index];
                CTbyte axis = m_splits_Axis[s.index];
                CTuint below = m_splits_Below[s.index];
                CTuint above = m_splits_Above[s.index];
                __ct_printf("contentCount=%d axis=%d plane=%f sah=%f below=%d above=%d\n", cc, (CTuint)axis, plane, s.sah, below, above);

                for(int a = start; a < start + length; ++a)
                {
                    std::stringstream ss;
                    ss << m_nodesBBox[0][i];
                        __ct_printf("%d [%d %d] id=%d Axis=%d, Plane=%f SAH=%f :: %s\n", 
                            a, m_splits_Below[a], m_splits_Above[a],
                            m_splits_IndexedSplit[a].index, 
                            (CTuint)m_splits_Axis[a],
                            m_splits_Plane[a], 
                            (m_splits_IndexedSplit[a].sah == INVALID_SAH ? -1 : m_splits_IndexedSplit[a].sah), ss.str().c_str());
                }

            }

            start += length;
        }
#endif
    }
    else
    {
        //m_dthAsyncNodesContent.WaitForCopy();
        m_hNodesContentCount.Resize(nodeCount);
        nutty::Copy(m_hNodesContentCount.Begin(), m_nodes_ContentCount.Begin(), nodeCount);

        for(int i = 0; i < nodeCount; ++i)
        {
            CTuint cc = m_hNodesContentCount[i];
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
        
//             nutty::cuStream& stream = m_pool.PeekNextStream();
//             nutty::SetStream(stream);

            nutty::Reduce(m_splits_IndexedSplit.Begin() + start, m_splits_IndexedSplit.Begin() + start + length, ReduceIndexedSplit(), neutralSplit, m_pStream);

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

//         for(CTuint i = 0; i < min(m_pool.GetStreamCount(), nodeCount); ++i)
//         {
//             nutty::cuStream& stream = m_pool.GetStream(i);
//             nutty::cuEvent e = stream.RecordEvent();
//             cudaStreamWaitEvent(0, e.GetPointer(), 0);
//         }
//     
//         nutty::SetDefaultStream();
    }
}

CTuint cuKDTreeScan::CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTbyte>::iterator& isLeafBegin, CTuint nodeOffset, CTuint nodeRange)
{
    m_leafCountScanner.Resize(nodeRange);

    m_leafCountScanner.ExcScan(isLeafBegin + nodeOffset, isLeafBegin + nodeOffset + nodeRange, TypeOp<CTbyte>(), m_pStream);

    DEVICE_SYNC_CHECK();

    m_dthAsyncIntCopy.WaitForStream(m_stream);
    m_dthAsyncByteCopy.WaitForStream(m_stream);

    m_dthAsyncIntCopy.StartCopy(m_leafCountScanner.GetPrefixSum().GetConstPointer() + nodeRange - 1, 0);
    m_dthAsyncByteCopy.StartCopy(isLeafBegin() + nodeOffset + nodeRange - 1, 0);

    CTuint block = NODES_GROUP_SIZE; //nutty::cuda::GetCudaBlock(nodeRange, 256U);
    CTuint grid = nutty::cuda::GetCudaGrid(nodeRange, block);

    DEVICE_SYNC_CHECK();

    if(m_interiorCountScanned.Size() <= nodeRange)
    {
        m_interiorCountScanned.Resize(nodeRange);
        m_maskedInteriorContent.Resize(nodeRange);
        m_interiorContentScanner.Resize(nodeRange);
        m_leafContentScanned.Resize(nodeRange);
    }

    createInteriorContentCountMasks<<<grid, block, 0, m_pStream>>>(
        isLeafBegin() + nodeOffset,
        m_nodes_ContentCount.Begin()(), 
        m_maskedInteriorContent.Begin()(), nodeRange);

    DEVICE_SYNC_CHECK();

    m_interiorContentScanner.ExcScan(m_maskedInteriorContent.Begin(), m_maskedInteriorContent.Begin() + nodeRange, nutty::PrefixSumOp<CTuint>(), m_pStream);
    
    DEVICE_SYNC_CHECK();

    makeOthers<<<grid, block, 0, m_pStream>>>(

        m_nodes_ContentStartAdd.Begin()(), 
        m_interiorContentScanner.GetPrefixSum().Begin()(), 
        m_leafContentScanned.Begin()(), 
        
        m_leafCountScanner.GetPrefixSum().Begin()(), 
        m_interiorCountScanned.Begin()(), 
        
        nodeRange);

    DEVICE_SYNC_CHECK();

    m_dthAsyncIntCopy.WaitForCopy();
    m_dthAsyncByteCopy.WaitForCopy();
    CTuint leafCount = m_dthAsyncIntCopy[0] + (m_dthAsyncByteCopy[0] == 1);

    DEVICE_SYNC_CHECK();

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

    m_dthAsyncIntCopy.StartCopy(m_leafContentScanned.GetConstPointer() + nodeCount - 1, 0);
    m_dthAsyncIntCopy.StartCopy(m_nodes_ContentCount.GetConstPointer() + nodeCount - 1, 1);
    m_dthAsyncByteCopy.StartCopy(m_activeNodesIsLeaf.GetConstPointer() + nodeCount + nodeOffset - 1, 0);
 
    m_leafNodesContentStart.Resize(currentLeafCount + leafCount);
    m_leafNodesContentCount.Resize(currentLeafCount + leafCount);

    const CTuint eventBlock = EVENT_GROUP_SIZE;
    CTuint eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);

    CTuint nodeBlock = NODES_GROUP_SIZE;
    CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);
    

#if 1
//     m_eventIsLeafScanner.Resize(eventCount);
//     m_eventIsLeafScanner.ExcScan(m_eventIsLeaf.Begin(), m_eventIsLeaf.Begin() + eventCount, TypeOp<CTbyte>());

    m_eventIsLeafScanned.Resize(eventCount);
    m_eventIsLeafScannedSums.Resize(eventCount/256 + 256);

    binaryGroupScan<256><<<eventGrid, eventBlock, 0, m_pStream>>>(
        m_eventIsLeaf.GetConstPointer(), m_eventIsLeafScanned.GetPointer(), m_eventIsLeafScannedSums.GetPointer(), TypeOp<CTbyte>(), eventCount);

    DEVICE_SYNC_CHECK();

    CTuint sumsCount = nutty::cuda::GetCudaGrid(eventCount, EVENT_GROUP_SIZE);

    if(sumsCount > 1)
    {
        nutty::PrefixSumOp<CTuint> _op;
        completeScan<256><<<1, 256, 0, m_pStream>>>(m_eventIsLeafScannedSums.GetConstPointer(), m_eventIsLeafScannedSums.GetPointer(), _op, sumsCount);

        DEVICE_SYNC_CHECK();

        spreadScannedSumsSingle<<<eventGrid-1, eventBlock, 0, m_pStream>>>(
                m_eventIsLeafScanned.GetPointer(), m_eventIsLeafScannedSums.GetConstPointer(), eventCount);
    }

#endif
    
    DEVICE_SYNC_CHECK();

    if(m_leafNodesContent.Size() < leafContentOffset + eventCount/2)
    {
        m_leafNodesContent.Resize(leafContentOffset + eventCount/2);
    }

    DEVICE_SYNC_CHECK();

    compactMakeLeavesData<<<eventGrid, eventBlock, 0, m_pStream>>>(
        isLeafBegin() + nodeOffset,
        m_interiorCountScanned.GetPointer(),
        m_leafContentScanned.GetPointer(),
         
        m_eventIsLeafScanned.GetConstPointer(),
        //m_eventIsLeafScanner.GetPrefixSum().GetConstPointer(),
            
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
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_nodes_ContentCount.GetPointer(), m_newNodesContentCount.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_nodes_ContentStartAdd.GetPointer(), m_newNodesContentStartAdd.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
        CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_activeNodes.GetPointer(), m_newActiveNodes.GetPointer(), nodeCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
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

struct ClipMaskPrefixSumOP
{
    __device__ CTuint operator()(CTbyte elem)
    {
        return isSet(elem) ? 1 : 0;
    }

    __device__ __host__ CTuint GetNeutral(void)
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
    ConstTuple<3, CTbyte> ptr;
    ptr.ts[0] = m_clipsMask.mask[0].GetConstPointer();
    ptr.ts[1] = m_clipsMask.mask[1].GetConstPointer();
    ptr.ts[2] = m_clipsMask.mask[2].GetConstPointer();

    Tuple<3, CTuint> ptr1;
    ptr1.ts[0] = m_clipsMask.scannedMask[0].GetPointer();
    ptr1.ts[1] = m_clipsMask.scannedMask[1].GetPointer();
    ptr1.ts[2] = m_clipsMask.scannedMask[2].GetPointer();
    
    Tuple<3, CTuint> sums;
    sums.ts[0] = m_clipsMask.scannedSums[0].GetPointer();
    sums.ts[1] = m_clipsMask.scannedSums[1].GetPointer();
    sums.ts[2] = m_clipsMask.scannedSums[2].GetPointer();

    ClipMaskPrefixSumOP op;
    ScanBinaryTriples(ptr, ptr1, sums, eventCount, op, m_pStream);

    //m_clipsMask.maskScanner[0].ExcScan(m_clipsMask.mask[0].Begin(), m_clipsMask.mask[0].Begin() + eventCount, op, m_pStream);
//     m_clipsMask.maskScanner[1].ExcScan(m_clipsMask.mask[1].Begin(), m_clipsMask.mask[1].Begin() + eventCount, op, m_pStream);
//     m_clipsMask.maskScanner[2].ExcScan(m_clipsMask.mask[2].Begin(), m_clipsMask.mask[2].Begin() + eventCount, op, m_pStream);

}

CT_RESULT cuKDTreeScan::Update(void)
{
    if(!m_initialized)
    {
        InitBuffer();
        m_initialized = true;
    }

    //ClearBuffer();
    
    CTuint primitiveCount = m_currentTransformedVertices.Size() / 3;

    static bool staticc = true;

    cudaCreateTriangleAABBs(m_currentTransformedVertices.GetPointer(), m_primAABBs.GetPointer(), primitiveCount, m_pStream);

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
        staticc = false;
    }

    DEVICE_SYNC_CHECK(); 
    
    CTuint elementBlock = EVENT_GROUP_SIZE;//nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    CTuint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);
    
    m_eventLines.Resize(2 * primitiveCount, m_pStream);

#ifdef PROFILE
        chimera::util::HTimer g_timer;
        cudaDeviceSynchronize();
        g_timer.Start();
        g_time = 0;
#endif

    m_eventLines.toggleIndex = 0;

    createEventsAndInit3<1, 0><<<elementGrid, elementBlock, 0, m_pStream>>>(
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
            EventSort(),
            m_pStream);
    }
    DEVICE_SYNC_CHECK();

    reorderEvent3<<<2 * elementGrid, elementBlock, 0, m_pStream>>>(2 * primitiveCount);

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
        static int i = 0;
        //__ct_printf("New Level=%d Events=%d (Frame=%d)\n", d, eventCount, ++i);
        
        CTuint nodeCount = g_interiorNodesCountOnThisLevel;
        CTuint nodeBlock = NODES_GROUP_SIZE;//nutty::cuda::GetCudaBlock(nodeCount, 256U);
        CTuint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        CTuint eventBlock = EVENT_GROUP_SIZE;//nutty::cuda::GetCudaBlock(eventCount, 256U);
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

        ComputeSAH_Splits(
            nodeCount, 
            eventCount,
            m_nodes_ContentCount.Begin()());

        DEVICE_SYNC_CHECK();

        makeLeafIfBadSplitOrLessThanMaxElements<<<nodeGrid, nodeBlock, 0, m_pStream>>>(
            m_nodes,
            m_nodes_IsLeaf.GetPointer() + g_nodeOffset,
            m_activeNodes.GetPointer(),
            m_activeNodesIsLeaf.GetPointer(), 
            m_splits,
            d == m_depth-1,
            nodeCount);

        DEVICE_SYNC_CHECK();

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

//              CTuint tb = 32;
//              CTuint tg = nutty::cuda::GetCudaGrid(count, tb);

            createClipMask<<<grid, block, 0, m_pStream>>>(
                m_nodes_ContentStartAdd.GetPointer(), 
                m_nodes_ContentCount.GetPointer(),
                count,
                m_eventLines.toggleIndex);

//             clipEvents3<<<grid, block, 0, m_pStream>>>(
//                 m_nodes_ContentStartAdd.GetPointer(), 
//                 m_nodes_ContentCount.GetPointer(),
//                 count,
//                 m_eventLines.toggleIndex);



            CTuint toggleSave = m_eventLines.toggleIndex;

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
            ScanClipMaskTriples(2 * count);
            //m_clipsMask.mask3Scanner.ExcScan(m_clipsMask.mask3.Begin(), m_clipsMask.mask3.Begin() + 2 * count, ClipMaskPrefixSum3OP());
// 
            m_dthAsyncIntCopy.WaitForStream(m_stream);
            m_dthAsyncByteCopy.WaitForStream(m_stream);

            //m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.scannedMask[0].GetConstPointer() + count - 1), 0);
            m_dthAsyncIntCopy.StartCopy((CTuint*)(m_clipsMask.scannedMask[0].GetConstPointer() + 2 * count - 1), 1);

            //m_dthAsyncByteCopy.StartCopy((CTbyte*)(m_clipsMask.mask[0].GetPointer() + count - 1), 0);
            m_dthAsyncByteCopy.StartCopy((CTbyte*)(m_clipsMask.mask[0].GetPointer() + 2 * count - 1), 1);

            CTuint _block = EVENT_GROUP_SIZE;
            CTuint _grid = nutty::cuda::GetCudaGrid(2 * count, block);

            compactEventLineV2<<<_grid, _block, 0, m_pStream>>>(
                2 * count,
                m_eventLines.toggleIndex);
            
            DEVICE_SYNC_CHECK();
            
            m_eventLines.Toggle();

            g_leafContentOffset += leavesRes.leafPrimitiveCount;

            if(lastLeaves)
            {
                setActiveNodesMask<1><<<nodeGrid, nodeBlock, 0, m_pStream>>>(
                    m_activeNodesThisLevel.Begin()(), 
                    m_activeNodesIsLeaf.Begin()(), 
                    m_interiorCountScanned.Begin()(),
                    0, 
                    nodeCount);
            }
            
            CTuint childCount = (nodeCount - leavesRes.leafCount) * 2;
            CTuint thisLevelNodesLeft = nodeCount - leavesRes.leafCount;
            
            nodeBlock = NODES_GROUP_SIZE;
            nodeGrid = nutty::cuda::GetCudaGrid(thisLevelNodesLeft, nodeBlock);

            initInteriorNodes<<<nodeGrid, nodeBlock, 0, m_pStream>>>(
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
                leavesRes.leafCount);

            CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_activeNodes.GetPointer(), m_newActiveNodes.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
            CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_nodes_ContentCount.GetPointer(), m_newNodesContentCount.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
            //CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_nodes_ContentStartAdd.GetPointer(), m_newNodesContentStartAdd.GetPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));

            DEVICE_SYNC_CHECK();

            m_dthAsyncIntCopy.WaitForCopy();
            m_dthAsyncByteCopy.WaitForCopy();
 
            //CTuint ccLeft = m_dthAsyncIntCopy[0] + isSet(m_dthAsyncByteCopy[0]);

            eventCount = m_dthAsyncIntCopy[1] + isSet(m_dthAsyncByteCopy[1]);// - ccLeft; //m_clipsMask.scannedMask[0][2 * count - 1] + isSet(m_clipsMask.mask[0][2 * count - 1]);

            if(eventCount == 0)
            {
                 __ct_printf("FATAL ERROR eventCount %d \n");
                exit(0);
            }

//             PrintBuffer(m_eventLines.eventLines[0].nodeIndex->Get(m_eventLines.toggleIndex), eventCount);
//             PrintBuffer(m_eventLines.eventLines[0].mask, 2 * count);
//             PrintBuffer(m_clipsMask.scannedMask[0], 2 * count);

            m_dthAsyncByteCopy.WaitForStream(m_stream);
            m_dthAsyncByteCopy.StartCopy(m_gotLeaves.GetConstPointer(), 0);
           
            eventBlock = EVENT_GROUP_SIZE;
            eventGrid = nutty::cuda::GetCudaGrid(eventCount, eventBlock);
            
            setEventsBelongToLeafAndSetNodeIndex<<<eventGrid, eventBlock, 0, m_pStream>>>(
                m_activeNodesIsLeaf.GetPointer() + nodeCount,
                m_eventIsLeaf.GetPointer(),
                m_nodes_NodeIdToLeafIndex.GetPointer() + g_childNodeOffset,
                eventCount,
                2 * nodeCount,
                m_eventLines.toggleIndex);

            DEVICE_SYNC_CHECK();

            //PROFILE_END;

            //if(!m_dthAsyncByteCopy[0])
            {
                m_interiorContentScanner.Resize(childCount);
                m_interiorContentScanner.ExcScan(m_nodes_ContentCount.Begin(), m_nodes_ContentCount.Begin() + childCount, nutty::PrefixSumOp<CTuint>());
                
                DEVICE_SYNC_CHECK();

                CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(
                    m_nodes_ContentStartAdd.GetPointer(), m_interiorContentScanner.GetPrefixSum().GetConstPointer(), childCount * sizeof(CTuint), cudaMemcpyDeviceToDevice, m_pStream));
                //nutty::Copy(m_nodes_ContentStartAdd.Begin(), m_interiorContentScanner.GetPrefixSum().Begin(), m_interiorContentScanner.GetPrefixSum().Begin() + childCount);

                DEVICE_SYNC_CHECK();    
            }

            m_dthAsyncByteCopy.WaitForCopy();

            DEVICE_SYNC_CHECK();
#if 0
            if(eventCount != 2 * (m_nodes_ContentCount[childCount - 1] + m_nodes_ContentStartAdd[childCount - 1]))
            {
                CTuint sum = 0;
                CTuint oldSum = 0;
                CTuint offset = 0;
                CTuint oldContentCount = 0;
                __ct_printf("nodeCount=%d\n", nodeCount);
                PrintBuffer(m_hNodesContentCount, nodeCount);
                PrintBuffer(m_lastNodeContentStartAdd, nodeCount);
                for(int i = 0; i < nodeCount; ++i)
                {
                    CTuint elemsLeft = 4 * m_lastNodeContentStartAdd[i];
                    CTuint contentCount = m_hNodesContentCount[i];
                    CTuint add = m_splits_IndexedSplit[oldSum].index;
                    CTbyte axis = m_splits_Axis[add];
                    CTuint above = m_splits_Above[add];
                    CTuint below = m_splits_Below[add];
                    oldSum += 2 * m_hNodesContentCount[i];

//                     if(IS_INVALD_SAH(m_splits_IndexedSplit[oldSum].sah))
//                     {
//                         __ct_printf("Invalid sah...\n");
//                     }
 
// 
                    //offset += 2 * below;
                    //aboveSum += 2 * above;
//                     //PrintBuffer(m_clipsMask.scannedMask[axis], 2 * oldContentLength + 2 * oldContentLength);
//                     //__ct_printf("\n");
//                     //PrintBuffer(m_clipsMask.mask[axis], 2 * oldContentLength + 2 * oldContentLength);
// 
//                     for(int i = 0; i < 2 * count - 1; ++i)
//                     {
//                         scannedMaskFile << m_clipsMask.scannedMask[axis][i] << "\n";
//                         maskFile << m_clipsMask.mask[axis][i] << "\n";
//                     }
// 
//                     scannedMaskFile.close();
//                     maskFile.close();
                    
                    //__ct_printf("\n%d %d %d\n", oldContentLength, 2 * oldContentLength + 2 * oldContentLength);
                    CTuint m0 = m_clipsMask.mask[axis][elemsLeft + 2 * contentCount - 1];
                    CTuint m1 = m_clipsMask.mask[axis][elemsLeft + 4 * contentCount - 1];

                    CTuint first = m_clipsMask.scannedMask[axis][elemsLeft + 2 * contentCount - 1];
                    CTuint second = m_clipsMask.scannedMask[axis][elemsLeft + 4 * contentCount - 1];

                    CTuint eventsLeft = first + (m0 > 0 ? 1 : 0) - m_clipsMask.scannedMask[axis][elemsLeft];

                    CTuint eventsRight = second + (m1 > 0 ? 1 : 0) - eventsLeft - m_clipsMask.scannedMask[axis][elemsLeft];

                    CTuint mask1 = 0; CTuint mask = 0;
//                     for(int i = elemsLeft; i < elemsLeft + 2 * contentCount; ++i)
//                     {
//                         mask1 += (m_clipsMask.mask[axis][i] != 0 ? 1 : 0);
// //                              scannedMaskFile << m_clipsMask.scannedMask[axis][i] << "\n";
// //                              maskFile << (m_clipsMask.mask[axis][i] != 0 ? 1 : 0) << "\n";
//                     }
// 

//                     for(int i = elemsLeft + 2 * contentCount; i < elemsLeft + 4 * contentCount; ++i)
//                     {
//                         mask += (m_clipsMask.mask[axis][i] != 0 ? 1 : 0);
// //                              scannedMaskFile << m_clipsMask.scannedMask[axis][i] << "\n";
// //                              maskFile << (m_clipsMask.mask[axis][i] != 0 ? 1 : 0) << "\n";
//                     }

                    __ct_printf("%d %d - %d %d - %d %d masks=%d %d, scan=%d elemsLeft=%d\n", 
                        2 * below, 2 * above, eventsLeft, eventsRight, mask, mask1, m0, m1, m_clipsMask.scannedMask[axis][2 * count-1], elemsLeft);
// 
// 
//                     oldContentCount = offset;
// 
                    // offset += 2 * below + 2 * above;
//                     std::ofstream maskFile("nodeMaskCreation.txt");
//                     CTuint ec = 0;
//                 CTuint srcAdd = toggleSave;
// 
//                     nutty::HostBuffer<IndexedSAHSplit> hisplit(count); nutty::Copy(hisplit.Begin(), m_splits_IndexedSplit.Begin(), m_splits_IndexedSplit.Begin() + count);
// 
//                     nutty::HostBuffer<CTuint> hContentStart(count); nutty::Copy(hContentStart.Begin(), m_lastNodeContentStartAdd.Begin(), m_lastNodeContentStartAdd.Begin() + count);
// 
//                     nutty::HostBuffer<CTbyte> hsplitAxis(count); nutty::Copy(hsplitAxis.Begin(), m_splits_Axis.Begin(), m_splits_Axis.Begin() + count);
// 
//                     nutty::HostBuffer<CTreal> hSplit(count); nutty::Copy(hSplit.Begin(), m_splits_Plane.Begin(), m_splits_Plane.Begin() + count);
// 
//                     nutty::HostBuffer<BBox> hbbox(count); 
//                     nutty::Copy(hbbox.Begin(), m_eventLines.eventLines[axis].ranges[srcAdd].Begin(), m_eventLines.eventLines[axis].ranges[srcAdd].Begin() + count);
// 
//                     nutty::HostBuffer<IndexedEvent> hv(count); 
//                     nutty::Copy(hv.Begin(), m_eventLines.eventLines[axis].indexedEvent[srcAdd].Begin(), m_eventLines.eventLines[axis].indexedEvent[srcAdd].Begin() + count);
// 
//                     nutty::HostBuffer<CTbyte> htype(count); nutty::Copy(htype.Begin(), m_eventLines.eventLines[axis].type[srcAdd].Begin(), m_eventLines.eventLines[axis].type[srcAdd].Begin() + count);
// 
//                     for(int id = 0; id < count; ++id)
//                     {
//               
//                         CTuint nodeIndex = 0;//m_eventLines.eventLines[axis].nodeIndex->Get(srcAdd)[id];
//                         CTuint eventsLeftFromMe = 2 * hContentStart[nodeIndex];
//                         IndexedSAHSplit isplit = hisplit[eventsLeftFromMe];
//                         CTbyte splitAxis = hsplitAxis[isplit.index];
//                         CTreal split = hSplit[isplit.index];
//                         CTuint N = eventsLeftFromMe + 2 * m_hNodesContentCount[nodeIndex];
//   
//                         BBox bbox = hbbox[id];
//                         CTreal v = hv[id].v;    
//                         CTbyte type = htype[id];
// 
//                         CTreal minAxis = getAxis(bbox.m_min, splitAxis);
//                         CTreal maxAxis = getAxis(bbox.m_max, splitAxis);
//                         
//                         if(maxAxis <= split)
//                         {
//                             maskFile << "LEFT index=";
//                             ec++;
//                         }
//                         else if(minAxis >= split)
//                         {
//                             maskFile << "RIGHT index=";
//                             ec++;
//                         }
//                         else
//                         {
//                             maskFile << "CLIP index=";
//                             ec += 2;
//                         }
// 
//                         maskFile << isplit.index << " maxAxis=" << maxAxis << " minAxis=" << minAxis << " split=" << split << " v=" << v << " nodeIndex=" << nodeIndex << " splitAxis=" << splitAxis << " axis=" << axis << "\n";
//                     }
//                     maskFile.close();
// 
//                     __ct_printf("testcount=%d\n", ec);
                }

                __ct_printf("%d %d\n", d, childCount);

                __ct_printf("FATAL ERROR eventCount %d != %d\n", eventCount, 2 * (m_nodes_ContentCount[childCount - 1] + m_nodes_ContentStartAdd[childCount - 1]));

                __debugbreak();
            }
#endif
            leavesRes = MakeLeaves(
                m_activeNodesIsLeaf.Begin(),
                g_childNodeOffset, 
                nodeCount, 
                childCount,
                eventCount, 
                g_currentLeafCount + lastLeaves, 
                g_leafContentOffset, 1, 
                m_dthAsyncByteCopy[0]);

            DEVICE_SYNC_CHECK();

            const static CTbyte null = 0;
            CUDA_RT_SAFE_CALLING_SYNC(cudaMemcpyAsync(m_gotLeaves.GetPointer(), &null, sizeof(CTbyte), cudaMemcpyHostToDevice, m_pStream));
            
            DEVICE_SYNC_CHECK();

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
            cudaMemcpyAsync(m_nodesBBox[0].GetPointer(), m_nodesBBox[1].GetConstPointer(), g_interiorNodesCountOnThisLevel * sizeof(BBox), cudaMemcpyDeviceToDevice, m_pStream);
            //nutty::Copy(m_nodesBBox[0].Begin(), m_nodesBBox[1].Begin(), m_nodesBBox[1].Begin() + g_interiorNodesCountOnThisLevel);
        }

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

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaStreamSynchronize(m_pStream));

//     ct_printf("Tree Summary:\n");
//     PRINT_BUFFER(m_nodes_IsLeaf);
//     PRINT_BUFFER(m_nodes_Split);
//     PRINT_BUFFER(m_nodes_SplitAxis);
//     PRINT_BUFFER(m_nodes_LeftChild);
//     PRINT_BUFFER(m_nodes_RightChild);
//     PRINT_BUFFER(m_leafNodesContentCount);
//     PRINT_BUFFER(m_leafNodesContentStart);
//     PRINT_BUFFER(m_nodes_NodeIdToLeafIndex);

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