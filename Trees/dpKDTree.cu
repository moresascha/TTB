
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif

#include "cuKDTree.h"
#include "shared_kernel.h"
#include "shared_types.h"
#include <Reduce.h>
#include <Sort.h>
#include <Scan.h>
#include <ForEach.h>
#include <Fill.h>
#include <Functions.h>
#include <cuda/Globals.cuh>
#include <cuda/cuda_helper.h>
#include "buffer_print.h"
#include "kd_kernel.cuh"

__constant__ cuDeviceHeap* g_dheap;

template <
    typename T
>
class DeviceOnlyBuffer
{
private:
    T* m_pMem;
    nutty::DevicePtr<T>::size_type m_size;

public:
    __device__ DeviceOnlyBuffer(size_t size) : m_size(size), m_pMem(0)
    {
        if(size != 0)
        {
            cudaMalloc(&m_pMem, sizeof(T) * size);
        }
    }

    __device__ nutty::DevicePtr<T> Begin(void)
    {
        return GetPtr();
    }

    __device__ nutty::DevicePtr<T> End(void)
    {
        return nutty::DevicePtr<T>(m_pMem + m_size);
    }

    __device__ T Front(void)
    {
        return m_pMem[0];
    }

    __device__ T Back(void)
    {
        return m_pMem[m_size - 1];
    }

    __device__ void Free(void)
    {
        if(m_pMem)
        {
            cudaFree(m_pMem);
            m_pMem = 0;
        }
    }

    __device__ T operator[](CTuint index)
    {
        return m_pMem[index];
    }

    __device__ ~DeviceOnlyBuffer(void)
    {
        Free();
    }

    __device__ size_t Size(void)
    {
        return m_size;
    }

    __device__ nutty::DevicePtr<T> GetPtr(void)
    {
        return nutty::DevicePtr<T>(m_pMem);
    }
};

struct dpSplit
{
    IndexedSAHSplit* indexedSplit;
    CTreal* v;
};

struct dpEvent
{
    CTbyte* type;
    IndexedEvent* indexedEvent;
    CTuint* primId;
};

struct dpNodeContent
{
    CTuint* primMemory;
    CTuint primCount;

    __device__ __host__ dpNodeContent(void)
    {

    }

    __device__ __host__ dpNodeContent(const dpNodeContent& cpy)
    {
        primMemory = cpy.primMemory;
    }

    __device__ void Init(CTuint count, CTuint id)
    {
        primCount = count;
        primMemory = (CTuint*)g_dheap->Alloc(id, count * sizeof(CTuint));
    }

    __device__ CTuint operator[](CTuint index)
    {
        return primMemory[index];
    }

    __device__ void Add(CTuint index, CTuint primId)
    {
        primMemory[index] = primId;
    }

};

std::ostream& operator<<(std::ostream &out, const dpNodeContent& t)
{
    out << "[pc=" << t.primCount << ", ptr=" << t.primMemory << "]";
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

void cudpKDTree::InitBuffer(void)
{
    CTuint primitiveCount = m_orginalVertices.Size() / 3;

    m_depth = (byte)min(31, max(1, (m_depth == 0xFF ? GenerateDepth(primitiveCount) : m_depth)));

    CTuint maxInteriorNodesCount = elemsBeforeLevel(m_depth);

    CTuint maxLeafNodesCount = elemsOnLevel(m_depth);

    CTuint maxNodeCount = maxInteriorNodesCount + maxLeafNodesCount;

    m_deviceHeap.Init(maxNodeCount - 1);

    m_nodesBBox.Resize(maxInteriorNodesCount);
    m_nodesIsLeaf.Resize(maxNodeCount);
    m_nodesSplit.Resize(maxInteriorNodesCount);
    m_nodesStartAdd.Resize(maxNodeCount);
    m_nodesSplitAxis.Resize(maxInteriorNodesCount);

    m_nodesContentCount.Resize(maxNodeCount);

    m_nodes.aabb = m_nodesBBox.GetDevicePtr()();
    m_nodes.isLeaf = m_nodesIsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodesSplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodesSplit.GetDevicePtr()();
    m_nodes.contentStart = m_nodesStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodesContentCount.GetDevicePtr()();

    m_primAABBs.Resize(primitiveCount); nutty::ZeroMem(m_primAABBs);
    m_sceneBBox.Resize(1); nutty::ZeroMem(m_sceneBBox);

    ClearBuffer();
}

void cudpKDTree::ClearBuffer(void)
{
    nutty::ZeroMem(m_nodesBBox);
    nutty::ZeroMem(m_nodesContentCount);
    nutty::ZeroMem(m_nodesIsLeaf);
    nutty::ZeroMem(m_nodesSplit);
    nutty::ZeroMem(m_nodesStartAdd);
    nutty::ZeroMem(m_nodesSplitAxis);
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

    __device__ T GetNeutral(void)
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

    __device__ T GetNeutral(void)
    {
        return 0;
    }
};

__global__ void createEventsDP(
        CTuint nodeIndex,
        dpEvent events,
        BBox* primAxisAlignedBB,
        CTbyte axis,
        dpNodeContent* nodesContent,
        byte depth, 
        uint N)
{
    RETURN_IF_OOB(N);
    
    dpNodeContent& content = nodesContent[nodeIndex];

    CTuint primIndex = content.primMemory[id];

    BBox aabbs = primAxisAlignedBB[primIndex];
            
    uint start_index = 2 * id + 0;
    uint end_index = 2 * id + 1;

    events.indexedEvent[start_index].index = start_index;
    events.type[start_index] = EDGE_START;
    events.primId[start_index] = primIndex;
    events.indexedEvent[start_index].v = getAxis(aabbs._min, axis);
            
    events.indexedEvent[end_index].index = end_index;
    events.type[end_index] = EDGE_END;
    events.primId[end_index] = primIndex;
    events.indexedEvent[end_index].v = getAxis(aabbs._max, axis);
}

__global__ void classifyEdgesDP(
        CTuint nodeIndex,
        Node nodes,
        dpEvent events,
        CTreal split,
        uint* edgeMask,
        uint N)
{
    RETURN_IF_OOB(N);

    CTreal v = events.indexedEvent[id].v;

    CTbyte type = events.type[id];

    CTint right = !(split > v || (v == split &&  type == EDGE_END));
            
    if(right)
    {
        edgeMask[id] = type == EDGE_START ? 0 : 1;
    }
    else
    {
        edgeMask[id] = type == EDGE_START ? 1 : 0;
    }
}

template <typename Buffer, typename T>
__device__ void Zero(Buffer& b)
{
    if(b.Size())
    {
        cudaMemsetAsync(b.Begin()(), 0, b.Size() * sizeof(T));
    }
}

__global__ void compactContent(CTuint* left, CTuint* right, CTuint below, dpEvent events, CTuint* mask, CTuint* scannedMask, CTuint N)
{
    RETURN_IF_OOB(N);

    if(mask[id] != 0)
    {
        CTuint dst = scannedMask[id];
        if(dst < below)
        {
            left[dst] = events.primId[id];
        }
        else
        {
            right[dst - below] = events.primId[id];
        }
    }
}

__global__ void computeSAHSplitsDP(
        CTuint nodeIndex,
        dpEvent events, 
        Node nodes,
        dpSplit splits,
        uint* scannedEdgeTypeStartMask,
        uint* scannedEdgeTypeEndMask,
        byte depth, 
        uint N) 
{
    RETURN_IF_OOB(N);

    CTbyte type = events.type[id];

    CTreal split = events.indexedEvent[id].v;

    CTuint primCount = nodes.contentCount[nodeIndex];

    BBox bbox = nodes.aabb[nodeIndex];

    CTbyte axis = getLongestAxis(bbox._min, bbox._max);

    CTuint above = primCount - scannedEdgeTypeStartMask[id];
    CTuint below = scannedEdgeTypeEndMask[id] - (type ^ 1);

    splits.indexedSplit[id].index = id;
    splits.indexedSplit[id].sah = getSAH(nodes.aabb[nodeIndex], axis, split, below, above);
    splits.v[id] = split;
}

__global__ void reorderEdgesDP(dpEvent dst, dpEvent src, uint N)
{
    RETURN_IF_OOB(N);
    CTuint index = src.indexedEvent[id].index;
    ONE_TO_ONE_CPY(primId);
    ONE_TO_ONE_CPY(type);
}

__global__ void buildKDTree(
    CTuint nodeIndex,
    CTuint maxDepth,
    Node nodes,
    BBox* primAxisAlignedBB, 
    CTuint primitiveCount, 
    CTuint depth,
    dpNodeContent* nodesContent,
    cudaError_t* error
    )
{   
    uint elementBlock = nutty::cuda::GetCudaBlock(primitiveCount, 256U);
    uint elementGrid = nutty::cuda::GetCudaGrid(primitiveCount, elementBlock);

    uint edgeCount = 2 * primitiveCount;
    uint edgeBlock = nutty::cuda::GetCudaBlock(edgeCount, 256U);
    uint edgeGrid = nutty::cuda::GetCudaGrid(edgeCount, edgeBlock);

    //memory start

    dpSplit splits;
    DeviceOnlyBuffer<IndexedSAHSplit> indexedSplits(edgeCount);
    DeviceOnlyBuffer<CTreal> splitPlanes(edgeCount);
    splits.indexedSplit = indexedSplits.Begin()();
    splits.v = splitPlanes.Begin()();

    DeviceOnlyBuffer<CTbyte> srcEventType(edgeCount);
    DeviceOnlyBuffer<IndexedEvent> indexedEvent(edgeCount);
    DeviceOnlyBuffer<CTuint> srcEventPrimId(edgeCount);

    dpEvent srcEvents;
    srcEvents.indexedEvent = indexedEvent.Begin()();
    srcEvents.type = srcEventType.Begin()();
    srcEvents.primId = srcEventPrimId.Begin()();

    DeviceOnlyBuffer<CTbyte> dstEventType(edgeCount);
    DeviceOnlyBuffer<CTuint> dstEventPrimId(edgeCount);

    dpEvent dstEvents;
    dstEvents.indexedEvent = indexedEvent.Begin()();
    dstEvents.type = dstEventType.Begin()();
    dstEvents.primId = dstEventPrimId.Begin()();

    DeviceOnlyBuffer<uint> scannedEdgeTypeStartMask(edgeCount);
    DeviceOnlyBuffer<uint> scannedEdgeTypeEndMask(edgeCount);
    DeviceOnlyBuffer<uint> scanMaskSums(edgeCount);

    //memory end

    BBox bbox = nodes.aabb[nodeIndex];
    CTuint axis = getLongestAxis(bbox._min, bbox._max);

    createEventsDP<<<elementGrid, elementBlock>>>(nodeIndex, srcEvents, primAxisAlignedBB, axis, nodesContent, depth, primitiveCount);

    nutty::Sort(indexedEvent.Begin(), indexedEvent.End(), EventSort());
    
    reorderEdgesDP<<<edgeGrid, edgeBlock>>>(dstEvents, srcEvents, edgeCount);
  
    EdgeTypeOp<CTbyte> op0;
    
    Zero<DeviceOnlyBuffer<uint>, uint>(scannedEdgeTypeEndMask);
    Zero<DeviceOnlyBuffer<uint>, uint>(scannedEdgeTypeStartMask);
    Zero<DeviceOnlyBuffer<uint>, uint>(scanMaskSums);

    *error = cudaDeviceSynchronize();

    nutty::InclusiveScan(dstEventType.Begin(), dstEventType.End(), scannedEdgeTypeStartMask.Begin(), scanMaskSums.Begin(), op0);
    
    cudaDeviceSynchronize();

    Zero<DeviceOnlyBuffer<uint>, uint>(scanMaskSums);

    cudaDeviceSynchronize();
    
    InvEdgeTypeOp<CTbyte> op1;

    nutty::InclusiveScan(dstEventType.Begin(), dstEventType.End(), scannedEdgeTypeEndMask.Begin(), scanMaskSums.Begin(), op1);
 
    computeSAHSplitsDP<<<edgeGrid, edgeBlock>>>(nodeIndex, dstEvents, nodes, splits, scannedEdgeTypeStartMask.Begin()(), scannedEdgeTypeEndMask.Begin()(), depth, edgeCount);
 
    IndexedSAHSplit neutralSplit;
    neutralSplit.index = 0;
    neutralSplit.sah = FLT_MAX;
    
    nutty::ReduceDP(indexedSplits.Begin(), indexedSplits.End(), ReduceIndexedSplit(), neutralSplit);
    
    cudaDeviceSynchronize();

    IndexedSAHSplit split = splits.indexedSplit[0];

    if(split.sah == FLT_MAX)
    {
        nodes.isLeaf[nodeIndex] = 1;
        return;
    }

    CTreal s = splits.v[split.index];
    CTbyte type = dstEventType[split.index];
    CTuint above = primitiveCount - scannedEdgeTypeStartMask[split.index];
    CTuint below = scannedEdgeTypeEndMask[split.index] - (type ^ 1);
    
    cudaMemsetAsync(scannedEdgeTypeEndMask.Begin()(), 0, scannedEdgeTypeEndMask.Size() * sizeof(uint));
    cudaMemsetAsync(scannedEdgeTypeStartMask.Begin()(), 0, scannedEdgeTypeStartMask.Size() * sizeof(uint));
    if(scanMaskSums.Size())
    {
        cudaMemsetAsync(scanMaskSums.Begin()(), 0, scanMaskSums.Size() * sizeof(uint));
    }

    cudaDeviceSynchronize();

    classifyEdgesDP<<<edgeGrid, edgeBlock>>>(nodeIndex, nodes, dstEvents, s, scannedEdgeTypeEndMask.Begin()(), edgeCount);
    nutty::ExclusivePrefixSumScan(scannedEdgeTypeEndMask.Begin(), scannedEdgeTypeEndMask.End(), scannedEdgeTypeStartMask.Begin(), scanMaskSums.Begin());
    
    cudaDeviceSynchronize();

    //BFO
    CTuint leftChildIndex = 2 * nodeIndex + 1;
    CTuint rightChildIndex = leftChildIndex + 1;

    nodes.contentCount[leftChildIndex] = below;
    nodes.contentCount[rightChildIndex] = above;

    dpNodeContent& c0c = nodesContent[leftChildIndex];
    dpNodeContent& c1c = nodesContent[rightChildIndex];

    c0c.Init(below, 2 * nodeIndex);
    c1c.Init(above, 2 * nodeIndex + 1);

    compactContent<<<edgeGrid, edgeBlock>>>(
        c0c.primMemory, c1c.primMemory, below, dstEvents, 
        scannedEdgeTypeEndMask.Begin()(), scannedEdgeTypeStartMask.Begin()(), edgeCount);

    cudaDeviceSynchronize();

    nodes.split[nodeIndex] = s;
    nodes.splitAxis[nodeIndex] = axis;

    *error = cudaDeviceSynchronize();

    if(*error != cudaSuccess)
    {
        return;
    }

    if(depth < maxDepth-1)
    {
        BBox l;
        BBox r;
        splitAABB(&nodes.aabb[nodeIndex], s, axis, &l, &r);
        nodes.aabb[leftChildIndex] = l;
        nodes.aabb[rightChildIndex] = r;

        if(below > MAX_ELEMENTS_PER_LEAF)
        {
            buildKDTree<<<1,1>>>(leftChildIndex, maxDepth, nodes, primAxisAlignedBB, below, depth + 1, nodesContent, error);
        }
        else
        {            
            nodes.isLeaf[leftChildIndex] = 1;
        }
        
        if(above > MAX_ELEMENTS_PER_LEAF)
        {
            buildKDTree<<<1,1>>>(rightChildIndex, maxDepth, nodes, primAxisAlignedBB, above, depth + 1, nodesContent, error);
        }
        else
        {
            nodes.isLeaf[rightChildIndex] = 1;
        }
    }
    else
    {
        nodes.isLeaf[leftChildIndex] = 1;
        nodes.isLeaf[rightChildIndex] = 1;
    }
}

cudpKDTree::cudpKDTree(void)
{
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 200 * 1024 * 1024)); //0.2GB
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 24));
}

CT_RESULT cudpKDTree::Update(void)
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

    BBox bboxN;
    bboxN._min = max3f;
    bboxN._max = min3f;

    nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), ReduceBBox(), bboxN);
    
    DEVICE_SYNC_CHECK();

    nutty::Copy(m_nodesBBox.Begin(), m_sceneBBox.Begin(), 1);

    m_nodesIsLeaf.Insert(0, m_depth == 0);
    
    nutty::DeviceBuffer<cudaError_t> error(1);
    error.Insert(0, cudaSuccess);
    cudaError_t e = error[0];
    CUDA_RT_SAFE_CALLING_NO_SYNC(e);

    dpNodeContent nc;
    ZeroMemory(&nc, sizeof(dpNodeContent));

    nutty::DeviceBuffer<dpNodeContent> dpNC(m_nodesContentCount.Size(), nc);
    nutty::DeviceBuffer<CTuint> d_rootNodeContent(primitiveCount);
    nutty::HostBuffer<CTuint> content(primitiveCount);
    nutty::Fill(content.Begin(), content.End(), nutty::unary::Sequence<CTuint>());
    nutty::Copy(d_rootNodeContent.Begin(), content.Begin(), content.End());

    nc.primCount = primitiveCount;
    nc.primMemory = d_rootNodeContent.GetDevicePtr()();
    dpNC.Insert(0, nc);

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyToSymbol(g_dheap, m_deviceHeap.GetDevPtr(), sizeof(cuDeviceHeap*)));

    size_t size;
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    ct_printf("size=%d\n", size);
    cudaDeviceGetLimit(&size, cudaLimitDevRuntimeSyncDepth);
    ct_printf("maxdepth=%d\n", size);

    buildKDTree<<<1,1>>>(0, m_depth, m_nodes, m_primAABBs.Begin()(), primitiveCount, 0, dpNC.Begin()(), error.Begin()());

    return CT_SUCCESS;
    e = error[0];
    CUDA_RT_SAFE_CALLING_NO_SYNC(e);

    PRINT_BUFFER(dpNC);

    m_deviceHeap.Print();

    for(int j = 0; j < m_deviceHeap.GetActiveBlocks(); ++j)
    {
        nutty::HostBuffer<CTuint> h(m_deviceHeap.GetBlockSize(j)/sizeof(CTuint));

        m_deviceHeap.Copy(h, j);

        for(int i = 0; i < h.Size(); ++i)
        {
            ct_printf("%d ", h[i]);
        }
        if(!h.Size())
        {
            ct_printf("%d empty...", j);
        }
        ct_printf("\n");
    }

    PRINT_BUFFER(m_primAABBs);

    PRINT_BUFFER(m_nodesBBox);
    PRINT_BUFFER(m_nodesIsLeaf);
    PRINT_BUFFER(m_nodesSplit);
    PRINT_BUFFER(m_nodesSplitAxis);
 
    PRINT_BUFFER(m_nodesContentCount);
    PRINT_BUFFER(m_nodesStartAdd);
     
    return CT_SUCCESS;
}