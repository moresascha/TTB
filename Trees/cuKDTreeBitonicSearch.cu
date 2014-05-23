
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#include "cuKDTree.h"
#include "kd_kernel.cuh"
#include "shared_kernel.h"
#include "shared_types.h"
#include <Reduce.h>
#include <Sort.h>
#include <Scan.h>
#include <ForEach.h>
#include <Fill.h>
#include <Functions.h>
#include <cuda/Globals.cuh>
#include "buffer_print.h"

std::ostream& operator<<(std::ostream &out, const IndexedSAHSplit& t)
{
    out << "SAH="<< (t.sah == FLT_MAX ? -1.0f : t.sah);//"[" << (t.sah == FLT_MAX ? -1.0f : t.sah) << ", " << t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const IndexedEvent& t)
{
    out << "Split=" << t.v;//"[" << t.v << ", " << t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const BBox& t)
{
    out << "[" << t._min.x << "," << t._min.y << "," << t._min.z << "|" << t._max.x << "," << t._max.y << "," << t._max.z << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const AABB& t)
{
    out << "[" << t.GetMin().x << "," << t.GetMin().y << "," << t.GetMin().z << "|" << t.GetMax().x << "," << t.GetMax().y << "," << t.GetMax().z << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const CTbyte& t)
{
    out << (CTuint)t;
    return out;
}

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

    CTuint maxInteriorNodesCount = elemsBeforeLevel(m_depth);

    CTuint maxLeafNodesCount = elemsOnLevel(m_depth);

    CTuint maxNodeCount = maxInteriorNodesCount + maxLeafNodesCount;

    m_nodesBBox.Resize(maxInteriorNodesCount);
    m_nodesIsLeaf.Resize(maxNodeCount);
    m_nodesSplit.Resize(maxInteriorNodesCount);
    m_nodesStartAdd.Resize(maxNodeCount);
    m_nodesSplitAxis.Resize(maxNodeCount);

    m_nodesContentCount.Resize(maxNodeCount);
    m_nodesAbove.Resize(maxNodeCount);
    m_nodesBelow.Resize(maxNodeCount);
    m_hNodesContentCount.Resize(elemsOnLevel(m_maxDepth-1));

    m_nodes.aabb = m_nodesBBox.GetDevicePtr()();
    m_nodes.isLeaf = m_nodesIsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodesSplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodesSplit.GetDevicePtr()();
    m_nodes.contentStart = m_nodesStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodesContentCount.GetDevicePtr()();
    m_nodes.below = m_nodesBelow.GetDevicePtr()();
    m_nodes.above = m_nodesAbove.GetDevicePtr()();

    /*m_bboxMin.Resize(m_primitiveCount); nutty::ZeroMem(m_bboxMin);
    m_bboxMax.Resize(m_primitiveCount); nutty::ZeroMem(m_bboxMax);*/

    m_primAABBs.Resize(primitiveCount); nutty::ZeroMem(m_primAABBs);
    m_sceneBBox.Resize(1); nutty::ZeroMem(m_sceneBBox);

    GrowMemory();

    ClearBuffer();
}

void cuKDTreeBitonicSearch::ClearBuffer(void)
{
    nutty::ZeroMem(m_edgeMask);
    nutty::ZeroMem(m_scannedEdgeMask);
    nutty::ZeroMem(m_edgeMaskSums);

    nutty::ZeroMem(m_nodesBBox);
    nutty::ZeroMem(m_nodesContentCount);
    nutty::ZeroMem(m_nodesIsLeaf);
    nutty::ZeroMem(m_nodesSplit);
    nutty::ZeroMem(m_nodesStartAdd);
    nutty::ZeroMem(m_nodesSplitAxis);
    nutty::ZeroMem(m_nodesAbove);
    nutty::ZeroMem(m_nodesBelow);

    nutty::ZeroMem(m_splitsAbove);
    nutty::ZeroMem(m_splitsBelow);
    nutty::ZeroMem(m_splitsAxis);
    nutty::ZeroMem(m_splitsSplit);

    m_edgesNodeIndex.ZeroMem();
    m_edgesPrimId.ZeroMem();
    m_edgesType.ZeroMem();
    m_edgesPrefixSum.ZeroMem();
}

void cuKDTreeBitonicSearch::GrowMemory(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;
    uint edgeCount = 4 * primitiveCount; //4 times as big
    m_primIndex.Resize(edgeCount);
    m_primNodeIndex.Resize(edgeCount);
    m_primPrefixSum.Resize(edgeCount);

    m_nodes.content = m_primIndex.GetDevicePtr()();

    m_nodesContent.primIndex = m_primIndex.GetDevicePtr()();
    m_nodesContent.nodeIndex = m_primNodeIndex.GetDevicePtr()();
    m_nodesContent.prefixSum = m_primPrefixSum.GetDevicePtr()();

    m_splitsAbove.Resize(edgeCount);
    m_splitsBelow.Resize(edgeCount);
    m_splitsAxis.Resize(edgeCount);
    m_splitsSplit.Resize(edgeCount);
    m_splitsIndexedSplit.Resize(edgeCount);
    
    m_splits.above = m_splitsAbove.GetDevicePtr()();
    m_splits.below = m_splitsBelow.GetDevicePtr()();
    m_splits.axis = m_splitsAxis.GetDevicePtr()();
    m_splits.indexedSplit = m_splitsIndexedSplit.GetDevicePtr()();
    m_splits.v = m_splitsSplit.GetDevicePtr()();

    m_edgesIndexedEdge.Resize(edgeCount);
    m_edgesNodeIndex.Resize(edgeCount);
    m_edgesPrimId.Resize(edgeCount);
    m_edgesType.Resize(edgeCount);
    m_edgesPrefixSum.Resize(edgeCount);

    m_edges[0].indexedEdge = m_edgesIndexedEdge.GetDevicePtr()();
    m_edges[0].nodeIndex = m_edgesNodeIndex.Get(0).GetDevicePtr()();
    m_edges[0].primId = m_edgesPrimId.Get(0).GetDevicePtr()();
    m_edges[0].type = m_edgesType.Get(0).GetDevicePtr()();
    m_edges[0].prefixSum = m_edgesPrefixSum.Get(0).GetDevicePtr()();

    m_edges[1].indexedEdge = m_edgesIndexedEdge.GetDevicePtr()();
    m_edges[1].nodeIndex = m_edgesNodeIndex.Get(1).GetDevicePtr()();
    m_edges[1].primId = m_edgesPrimId.Get(1).GetDevicePtr()();
    m_edges[1].type = m_edgesType.Get(1).GetDevicePtr()();
    m_edges[1].prefixSum = m_edgesPrefixSum.Get(1).GetDevicePtr()();

    m_edgeMask.Resize(edgeCount);
    m_scannedEdgeMask.Resize(edgeCount);
    m_edgeMaskSums.Resize(edgeCount); //way to big but /care
}


template <
    typename T
>
struct InvEdgeTypeOp
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
struct EdgeTypeOp
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

CT_RESULT cuKDTreeBitonicSearch::Update(void)
{

    if(!m_initialized)
    {
        InitBuffer();
        m_initialized = true;
    }

    ClearBuffer();

    static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
    static float3 min3f = -max3f;
    
    CTuint primitiveCount = m_currentTransformedVertices.Size() / 3;

    m_nodesContentCount.Insert(0, primitiveCount);

    uint _block = primitiveCount < 256 ? primitiveCount : 256;
    uint _grid = nutty::cuda::GetCudaGrid(primitiveCount, _block);
    
    cudaCreateTriangleAABBs(m_currentTransformedVertices.Begin()(), m_primAABBs.Begin()(), primitiveCount);
   
    DEVICE_SYNC_CHECK();

    initNodesContent<<<_grid, _block>>>(m_nodesContent, primitiveCount);

    DEVICE_SYNC_CHECK();

    BBox bboxN;
    bboxN._min = max3f;
    bboxN._max = min3f;

    nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), ReduceBBox(), bboxN);

    DEVICE_SYNC_CHECK();

    nutty::Copy(m_nodesBBox.Begin(), m_sceneBBox.Begin(), 1);

    m_nodesIsLeaf.Insert(0, m_depth == 0);

    for(byte d = 0; d < m_depth; ++d)
    {
        uint elementBlock = primitiveCount < 256 ? primitiveCount : 256;
        uint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);

//         nutty::ZeroMem(m_edgeMask);
//         nutty::ZeroMem(m_scannedEdgeMask);

        Edge edgesSrc = m_edges[0];
        Edge edgesDst = m_edges[1];
        
        DEVICE_SYNC_CHECK();

        createEdges<<<elementGrid, elementBlock>>>(edgesSrc, m_nodes, m_primAABBs.Begin()(), m_nodesContent, d, primitiveCount);

        DEVICE_SYNC_CHECK();

        uint copyStart = (1 << d) - 1;
        uint copyLength = 1 << d;

        nutty::Copy(m_hNodesContentCount.Begin(), m_nodesContentCount.Begin() + copyStart, copyLength);

        uint start = 0;

        uint nodeCount = 1 << d;
        uint nodeBlock = nodeCount < 256 ? nodeCount : 256;
        uint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

        for(int i = 0; i < (1 << d); ++i)
        {
            uint length = 2 * m_hNodesContentCount[i];
            if(length == 0)
            {
                continue;
            }

            nutty::Sort(m_edgesIndexedEdge.Begin() + start, m_edgesIndexedEdge.Begin() + start + length, EdgeSort());
            
            DEVICE_SYNC_CHECK();

            start += length;
        }

        DEVICE_SYNC_CHECK();

        uint edgeCount = 2 * primitiveCount;
        uint edgeBlock = edgeCount < 256 ? edgeCount : 256;
        uint edgeGrid = nutty::cuda::GetCudaGrid(edgeCount, edgeBlock);

        reorderEdges<<<edgeGrid, edgeBlock>>>(edgesDst, edgesSrc, edgeCount);

        m_scannedEdgeTypeStartMask.Resize(m_edgesType[1].Size());
        m_edgeTypeMaskSums.Resize(m_edgesType[1].Size());
        m_scannedEdgeTypeEndMask.Resize(m_edgesType[1].Size());

        EdgeTypeOp<CTbyte> op0;

        nutty::InclusiveScan(m_edgesType[1].Begin(), m_edgesType[1].Begin() + edgeCount, m_scannedEdgeTypeStartMask.Begin(), m_edgeTypeMaskSums.Begin(), op0);

        nutty::ZeroMem(m_edgeTypeMaskSums);
        nutty::ZeroMem(m_scannedEdgeTypeEndMask);

        InvEdgeTypeOp<CTbyte> op1;
        nutty::InclusiveScan(m_edgesType[1].Begin(), m_edgesType[1].Begin() + edgeCount, m_scannedEdgeTypeEndMask.Begin(), m_edgeTypeMaskSums.Begin(), op1);

        DEVICE_SYNC_CHECK();
//         PrintBuffer(m_edgesType[1], edgeCount);
//         PrintBuffer(m_scannedEdgeTypeStartMask, edgeCount);
//         PrintBuffer(m_scannedEdgeTypeEndMask, edgeCount);

        computeSAHSplits<<<edgeGrid, edgeBlock>>>(edgesDst, m_nodes, m_splits, m_nodesContent, m_scannedEdgeTypeStartMask.Begin()(), m_scannedEdgeTypeEndMask.Begin()(), d, edgeCount);

        //PrintBuffer(m_edgesIndexedEdge);
        //PrintBuffer(m_splitsIndexedSplit);

        for(int i = 0; i < edgeCount; ++i)
        {
            CTbyte t = m_edgesType[1][i] ^ 1;
            ct_printf("[%d %d] [%d %d] Split=%.4f SAH=%.4f\n", m_splitsBelow[i], m_splitsAbove[i], m_scannedEdgeTypeEndMask[i] - t, edgeCount/2 - m_scannedEdgeTypeStartMask[i], m_edgesIndexedEdge[i].v, m_splitsIndexedSplit[i].sah);
        }
        ct_printf("\n\n");
        DEVICE_SYNC_CHECK();
        
        start = 0;

        for(int i = 0; i < (1 << d); ++i)
        {
            uint length = 2 * m_hNodesContentCount[i];
            if(length == 0)
            {
                continue;
            }

            IndexedSAHSplit neutralSplit;
            neutralSplit.index = 0;
            neutralSplit.sah = FLT_MAX;
            
            nutty::Reduce(m_splitsIndexedSplit.Begin() + start, m_splitsIndexedSplit.Begin() + start + length, ReduceIndexedSplit(), neutralSplit);
            DEVICE_SYNC_CHECK();

            start += length;
        }

        if(d == m_depth-1)
        {
            initNodes<1><<<nodeGrid, nodeBlock>>>(m_nodes, m_splits, m_scannedEdgeTypeStartMask.Begin()(), d);
        }
        else
        {
            initNodes<0><<<nodeGrid, nodeBlock>>>(m_nodes, m_splits, m_scannedEdgeTypeStartMask.Begin()(), d);
        }

        DEVICE_SYNC_CHECK();

        classifyEdges<<<edgeGrid, edgeBlock>>>(m_nodes, edgesDst, m_edgeMask.Begin()(), d, edgeCount);

        DEVICE_SYNC_CHECK();

        nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.Begin() + edgeCount + 1, m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());
        //PrintBuffer(m_edgeMask);
       // PrintBuffer(m_scannedEdgeMask);
        DEVICE_SYNC_CHECK();
        
//         PrintBuffer(m_nodesContentCount);
//         if(d == m_depth-1)
//         {
//             initCurrentNodesAndCreateChilds<1><<<nodeGrid, nodeBlock>>>(m_scannedEdgeMask.Begin()(), m_edgeMask.Begin()(), m_nodes, d);
//         }
//         else
//         {
//             initCurrentNodesAndCreateChilds<0><<<nodeGrid, nodeBlock>>>(m_scannedEdgeMask.Begin()(), m_edgeMask.Begin()(), m_nodes, d);
//         }
    
        //PrintBuffer(m_nodesContentCount);
       // PrintBuffer(m_nodesStartAdd);
        
        DEVICE_SYNC_CHECK();

        //PrintBuffer(m_primIndex);

        compactContentFromEdges<<<edgeGrid, edgeBlock>>>(edgesDst, m_nodes, m_nodesContent, m_edgeMask.Begin()(), m_scannedEdgeMask.Begin()(), d, edgeCount);

        DEVICE_SYNC_CHECK();

        //PrintBuffer(m_primIndex);

        uint lastCnt = primitiveCount;
        primitiveCount = m_scannedEdgeMask[edgeCount - 1] + m_edgeMask[edgeCount - 1];

        if(primitiveCount == 0)
        {
            primitiveCount = lastCnt;
            break;
        }

        if(2 * primitiveCount > m_edgeMask.Size() && d < m_depth-1)
        {
            GrowMemory();
        }

        DEVICE_SYNC_CHECK();
    }

    //if(m_currentTransformedPrimitives.Size() < primitiveCount)
    {
        m_kdPrimitives.Resize(primitiveCount);
        //m_tPrimAABBs.Resize(primitiveCount);
    }

    uint block = primitiveCount < 256 ? primitiveCount : 256;
    uint grid = nutty::cuda::GetCudaGrid(primitiveCount, block);

    postprocess<<<grid, block>>>(m_currentTransformedVertices.Begin()(), m_kdPrimitives.Begin()(), m_nodesContent, primitiveCount);
    
    DEVICE_SYNC_CHECK();
 
    PrintBuffer(m_nodesBBox);
    PrintBuffer(m_nodesIsLeaf);
    PrintBuffer(m_nodesSplit);
    PrintBuffer(m_nodesSplitAxis);

    PrintBuffer(m_nodesContentCount);
    PrintBuffer(m_nodesStartAdd);

    PrintBuffer(m_primIndex);
    //PrintBuffer(m_primNodeIndex);
   // PrintBuffer(m_primPrefixSum);
     
    return CT_SUCCESS;
}