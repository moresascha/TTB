#pragma once

#include "tree.h"
#include <Nutty.h>
#include "geometry.h"
#include <DeviceBuffer.h>
#include <Functions.h>
#include <Copy.h>
#include <Scan.h>
#include <cuda/Stream.h>
#include <cuda/cuda_helper.h>
#include "memory.h"
#include "device_heap.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define RETURN_IF_OOB(__N) \
    CTuint id = GlobalId; \
    if(id >= __N) \
    { \
    return; \
    }

#define EVENT_START ((CTeventType_t)1)
#define EDGE_END    ((CTeventType_t)0)

template <
    typename T
>
struct ScanPrimitiveStruct
{
    __device__ T operator()(T elem)
    {
        return elem;
    }

    __device__ __host__ CTuint3 GetNeutral(void)
    {
        T v = {0};
        return v;
    }
};

template <
    typename T
>
class DoubleBuffer
{
private:
    nutty::DeviceBuffer<T> m_buffer[2];
    CTbyte m_current;

public:
    DoubleBuffer(void) :  m_current(0)
    {

    }

    void Resize(size_t size)
    {
        for(byte i = 0; i < 2; ++i)
        {
            m_buffer[i].Resize(size);
        }
    }

    nutty::DeviceBuffer<T>& Get(CTbyte index)
    {
        assert(index < 2);
        return m_buffer[index];
    }

    nutty::DeviceBuffer<T>& operator[](CTbyte index)
    {
        return Get(index);
    }

    nutty::DeviceBuffer<T>& GetCurrent(void)
    {
        return m_buffer[m_current];
    }

    nutty::Iterator<T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
    > Begin(CTbyte index)
    {
        return m_buffer[index].Begin();
    }

    nutty::Iterator<T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
    > End(CTbyte index)
    {
        return m_buffer[index].End();
    }

    void Toggle(void)
    {
        m_current = (m_current + 1) % 2;
    }

    size_t Size(void)
    {
        return m_buffer[0].Size();
    }

    void ZeroMem(void)
    {
        nutty::ZeroMem(m_buffer[0]);
        nutty::ZeroMem(m_buffer[1]);
    }
};

template <
    typename T
>
class cuAsyncDtHCopy
{
    T* m_pPitchedHostMemory;
    nutty::cuStream m_stream;
    cudaStream_t m_pStream;
    size_t m_slots;

public:
    cuAsyncDtHCopy(void) : m_pPitchedHostMemory(NULL), m_slots(0), m_pStream(NULL)
    {

    }

    cuAsyncDtHCopy(size_t slots) : m_pStream(NULL)
    {
        Init(slots);
    }

    void Init(size_t slots)
    {
        cudaMallocHost(&m_pPitchedHostMemory, sizeof(T) * slots, 0);
        m_slots = slots;
        m_pStream = m_stream();
    }

    void Resize(size_t slots)
    {
        if(slots <= m_slots)
        {
            return;
        }
        if(m_pPitchedHostMemory)
        {
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaFreeHost(m_pPitchedHostMemory));
        }
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMallocHost(&m_pPitchedHostMemory, sizeof(T) * slots, 0));
        m_slots = slots;
    }

    void WaitForStream(const nutty::cuStream& stream)
    {
        //cudaDeviceSynchronize();
        m_stream.WaitEvent(std::move(stream.RecordEvent()));
    }

//     void WaitForEvent(nutty::cuEvent event)
//     {
//         m_stream.WaitEvent(std::move(event));
//     }

    void StartCopy(const T* devSrc, size_t slot, size_t range = 1)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyAsync(m_pPitchedHostMemory + slot, devSrc, range * sizeof(T), cudaMemcpyDeviceToHost, m_pStream));
        //cudaMemcpy(m_pPitchedHostMemory + slot, devSrc, range * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void StartCopyUnsafe(const void* devSrc, size_t slot, size_t range = 1)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyAsync(m_pPitchedHostMemory + slot, devSrc, range, cudaMemcpyDeviceToHost, m_pStream));
        //cudaMemcpy(m_pPitchedHostMemory + slot, devSrc, range * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void WaitForCopy(void)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaStreamSynchronize(m_pStream));
        //cudaDeviceSynchronize();
       // m_stream.ClearEvents();
    }

    T operator[](size_t index)
    {
        //WaitForCopy();
        return m_pPitchedHostMemory[index];
    }

    ~cuAsyncDtHCopy(void)
    {
        if(m_pPitchedHostMemory)
        {
            CUDA_RT_SAFE_CALLING_SYNC(cudaFreeHost(m_pPitchedHostMemory));
        }
    }
};

__device__ __host__ __forceinline__ bool isSet(CTclipMask_t mask)
{
    return mask > 0;
}

__device__ __forceinline__ bool isLeft(CTclipMask_t mask)
{
    return (mask & 0x01) > 0;
}

__device__ __forceinline__ bool isRight(CTclipMask_t mask)
{
    return (mask & 0x02) > 0;
}

__device__ __forceinline__ bool isOLappin(CTclipMask_t mask)
{
    return (mask & 0x04) > 0;
}

__device__ __forceinline__ CTaxis_t getAxisFromMask(CTclipMask_t mask)
{
    mask = mask & 0xB8;
    switch(mask)
    {
    case 0x08 : return 0;
    case 0x10 : return 1;   
    case 0x20 : return 2;
    }
    return 0;
} 

__device__ __forceinline__ void setLeft(CTclipMask_t& mask)
{
    mask |= 0x01;
}

__device__ __forceinline__ void setRight(CTclipMask_t& mask)
{
    mask |= 0x02;
}

__device__ __forceinline__ void setOLappin(CTclipMask_t& mask)
{
    mask |= 0x04;
}

__device__ __forceinline__ void setAxis(CTclipMask_t& mask, CTaxis_t axis)
{
    mask |= axis == 0 ? 0x08 : axis == 1 ? 0x10 : 0x20;
}

struct IndexedEvent
{
    //CTuint index;
    CTreal v;
};

// struct Event
// {
//     CTbyte* type;
//     IndexedEvent* indexedEdge;
//     CTuint* nodeIndex;
//     CTuint* primId;
//     CTuint* prefixSum;
// };

struct IndexedSAHSplit
{
    CTreal sah;
    CTuint index;
};

struct Split
{
    IndexedSAHSplit* indexedSplit;
    CTaxis_t* axis;
    CTuint* below;
    CTuint* above;
    CTreal* v;
};

struct SplitConst
{
    const IndexedSAHSplit* __restrict indexedSplit;
    const CTaxis_t* __restrict axis;
    const CTuint* __restrict below;
    const CTuint* __restrict above;
    const CTreal* __restrict v;
};

struct Node
{
    CTnodeIsLeaf_t* isLeaf;
    CTreal* split;
    CTaxis_t* splitAxis;
    CTuint* contentCount;
    CTuint* contentStart;
    CTuint* leftChild;
    CTuint* rightChild;
    CTuint* nodeToLeafIndex;
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
        return elem; // ^ 1;
    }

    __device__ __host__ T GetNeutral(void)
    {
        return 0; //1
    }
};

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

struct EventSort
{
    __device__ char operator()(IndexedEvent t0, IndexedEvent t1)
    {
        return t0.v > t1.v;
    }
};

struct ReduceIndexedSplit
{
    __device__ __host__ IndexedSAHSplit operator()(IndexedSAHSplit t0, IndexedSAHSplit t1)
    {
        return t0.sah < t1.sah ? t0 : t1;
    }
};

class cuKDTree : public ICTTree
{
protected:
    nutty::cuStream m_stream;
    cudaStream_t m_pStream;

    ICTTreeNode* m_node;
    CT_GEOMETRY_TOPOLOGY m_topo;
    CTuint m_interiorNodesCount;
    CTuint m_leafNodesCount;
    CTbyte m_depth;
    CTuint m_flags;
    CTbool m_initialized;
    AABB m_sceneAABB;

    nutty::DeviceBuffer<CTreal3> m_orginalVertices;
    nutty::DeviceBuffer<CTreal3> m_currentTransformedVertices;

    //nutty::DeviceBuffer<CTreal3> m_kdPrimitives;

    nutty::DeviceBuffer<CTreal3> m_bboxMin;
    nutty::DeviceBuffer<CTreal3> m_bboxMax;
    nutty::DeviceBuffer<BBox> m_sceneBBox;

    nutty::DeviceBuffer<BBox> m_primAABBs;
    nutty::DeviceBuffer<BBox> m_tPrimAABBs;

    Node m_nodes;
    nutty::DeviceBuffer<CTnodeIsLeaf_t> m_nodes_IsLeaf;
    nutty::DeviceBuffer<CTaxis_t> m_nodes_SplitAxis;
    nutty::DeviceBuffer<CTreal> m_nodes_Split;
    nutty::DeviceBuffer<CTuint> m_nodes_ContentCount;
    nutty::DeviceBuffer<CTuint> m_nodes_ContentStartAdd;

    nutty::DeviceBuffer<CTuint> m_nodes_LeftChild;
    nutty::DeviceBuffer<CTuint> m_nodes_RightChild;
    nutty::DeviceBuffer<CTuint> m_nodes_NodeIdToLeafIndex;

    nutty::DeviceBuffer<CTuint> m_leafNodesContent;
    nutty::DeviceBuffer<CTuint> m_leafNodesContentCount;
    nutty::DeviceBuffer<CTuint> m_leafNodesContentStart;

    std::vector<CTulong> m_linearGeoHandles;
    std::map<CTGeometryHandle, GeometryRange> m_handleRangeMap;

    nutty::HostBuffer<CTaxis_t> m_host_nodes_SplitAxis;
    nutty::HostBuffer<CTreal> m_host_nodes_Split;
    nutty::HostBuffer<CTnodeIsLeaf_t> m_host_nodes_IsLeaf;
    nutty::HostBuffer<CTuint> m_host_nodes_LeftChild;
    nutty::HostBuffer<CTuint> m_host_nodes_RightChild;

    void _DebugDrawNodes(CTuint parent, const AABB& aabb, ICTTreeDebugLayer* dbLayer) const;

public:
    cuKDTree(void);

    void OnGeometryMoved(const CTGeometryHandle geo)
    {

    }

    cudaStream_t GetStream(void)
    {
        return m_pStream;
    }

    CT_RESULT RayCast(const CTreal3& eye, const CTreal3& dir, CTGeometryHandle* handle) const
    {
        return CT_NOT_YET_IMPLEMENTED;
    }

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo)
    {
        m_topo = topo;
        return CT_SUCCESS;
    }

    virtual CT_RESULT Init(CTuint flags);

    virtual CT_RESULT Update(void) = 0;

    ICTTreeNode* GetRoot(void) const
    {
        return m_node;
    }

    CT_RESULT DebugDraw(ICTTreeDebugLayer* dbLayer);

    CT_RESULT AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle);

    void SetDepth(CTbyte depth);

    CTuint GetDepth(void) const
    {
        return m_depth;
    }

    CTuint GetInteriorNodesCount(void) const
    {
        return m_interiorNodesCount;
    }
    
    CTuint GetLeafNodesCount(void) const
    {
        return m_leafNodesCount;
    }
    
    CT_TREE_DEVICE GetDeviceType(void) const
    {
        return eCT_GPU;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return m_topo;
    }

    const CTGeometryHandle* GetGeometry(CTuint* gc);

    const CTreal* GetRawPrimitives(CTuint* bytes) const;

    CTuint GetPrimitiveCount(void) const
    {
        return (CTuint)m_orginalVertices.Size() / 3;
    }

    CT_RESULT Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData = NULL);

    void TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix);

    virtual const void* GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const;

    CT_RESULT AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle);

    const ICTAABB* GetAxisAlignedBB(void) const
    {
        return &m_sceneAABB;
    }

    virtual ~cuKDTree(void);

    add_uuid_header(cuKDTree);
};

struct MakeLeavesResult
{
    CTuint leafPrimitiveCount;
    CTuint leafCount;
    CTuint interiorPrimitiveCount;
};

struct cuClipMask
{
    CTclipMask_t* mask;
    CTreal* newSplit;
    CTuint* index;

    //     __device__ void setLeft(CTuint id)
    //     {
    //         mask[id] |= 0x01;
    //     }
    // 
    //     __device__ void setRight(CTuint id)
    //     {
    //         mask[id] |= 0x02;
    //     }
    // 
    //     __device__ void setOLappin(CTuint id)
    //     {
    //         mask[id] |= 0x04;
    //     }
    // 
    //     __device__ void setAxis(CTuint id, CTbyte axis)
    //     {
    //         mask[id] |= axis == 0 ? 0x08 : axis == 1 ? 0x10 : 0x20;
    //     }
};

struct cuConstClipMask
{
    const CTclipMask_t* __restrict mask;
    const CTreal* __restrict newSplit;
    const CTuint* __restrict index;
    const CTuint* __restrict scanned;
    CTuint* elemsPerTile;
};

struct cuClipMaskArray
{
    cuClipMask mask[3];
    CTuint* scanned[3];
    CTuint* elemsPerTile[3];
};

struct ClipMask
{
    nutty::DeviceBuffer<CTclipMask_t> mask[3];
    nutty::DeviceBuffer<CTuint> scannedMask[3];
    nutty::DeviceBuffer<CTuint> scannedSums[3];

    nutty::DeviceBuffer<CTuint> elemsPerTile[3];

    //nutty::DeviceBuffer<CTbyte3> mask3;
    nutty::DeviceBuffer<CTreal> newSplits[3];
    nutty::DeviceBuffer<CTuint> index[3];
    //nutty::Scanner maskScanner[3];

    nutty::TScanner<CTuint3, ScanPrimitiveStruct<CTuint3>> mask3Scanner;

    void GetConstPtr(cuConstClipMask& mm, CTbyte i)
    {
        mm.mask = mask[i].GetConstPointer();

        mm.newSplit = newSplits[i].GetConstPointer();

        mm.index = index[i].GetConstPointer();

        //mm.scanned = maskScanner[i].GetPrefixSum().GetConstPointer();

        mm.elemsPerTile = elemsPerTile[i].GetPointer();

        mm.scanned = scannedMask[i].GetConstPointer();
    }

    void GetPtr(cuClipMaskArray& mm)
    {
        mm.mask[0].mask = mask[0].GetPointer();
        mm.mask[1].mask = mask[1].GetPointer();
        mm.mask[2].mask = mask[2].GetPointer();

        mm.mask[0].newSplit = newSplits[0].GetPointer();
        mm.mask[1].newSplit = newSplits[1].GetPointer();
        mm.mask[2].newSplit = newSplits[2].GetPointer();

        mm.mask[0].index = index[0].GetPointer();
        mm.mask[1].index = index[1].GetPointer();
        mm.mask[2].index = index[2].GetPointer();

        //         mm.scanned[0] = maskScanner[0].GetPrefixSum().GetConstPointer();//scannedMask[0].GetConstPointer();
        //         mm.scanned[1] = maskScanner[1].GetPrefixSum().GetConstPointer();//scannedMask[1].GetConstPointer();
        //         mm.scanned[2] = maskScanner[2].GetPrefixSum().GetConstPointer();//scannedMask[2].GetConstPointer();

        mm.scanned[0] = scannedMask[0].GetPointer();
        mm.scanned[1] = scannedMask[1].GetPointer();
        mm.scanned[2] = scannedMask[2].GetPointer();

        mm.elemsPerTile[0] = elemsPerTile[0].GetPointer();
        mm.elemsPerTile[1] = elemsPerTile[1].GetPointer();
        mm.elemsPerTile[2] = elemsPerTile[2].GetPointer();
    }

    void Resize(size_t size, cudaStream_t pStream);

    void ScanMasks(CTuint legnth);
};

struct cuEventLine
{
    IndexedEvent* indexedEvent;
    CTeventType_t* type;
    CTuint* nodeIndex;
    CTuint* primId;
    BBox* ranges;
    CTeventMask_t* mask;

    const CTuint* __restrict scannedEventTypeStartMask;
};

struct cuConstEventLine
{
    const IndexedEvent* indexedEvent;
    const CTeventType_t* __restrict type;
    const CTuint* __restrict nodeIndex;
    const CTuint* __restrict primId;
    const BBox* __restrict ranges;
    const CTeventMask_t* __restrict mask;

    const CTuint* __restrict scannedEventTypeStartMask;
};

struct EventLine
{
    DoubleBuffer<IndexedEvent> indexedEvent;

    thrust::device_vector<float>* rawEvents;
    thrust::device_vector<unsigned int>* eventKeys;

    DoubleBuffer<CTeventType_t> type;
    DoubleBuffer<CTuint> primId;
    DoubleBuffer<BBox> ranges;

    nutty::Scanner typeStartScanner;
    nutty::DeviceBuffer<CTuint> typeStartScanned;
    nutty::Scanner eventScanner;
    nutty::DeviceBuffer<CTuint> scannedEventTypeStartMask;
    nutty::DeviceBuffer<CTuint> scannedEventTypeStartMaskSums;
    nutty::DeviceBuffer<CTeventMask_t> mask;

    CTbyte toggleIndex;

    DoubleBuffer<CTuint>* nodeIndex;

    EventLine(void);

    void SetNodeIndexBuffer(DoubleBuffer<CTuint>* nodeIndex_)
    {
        nodeIndex = nodeIndex_;
    }

    void Resize(CTuint size);

    size_t Size(void);

    cuEventLine GetPtr(CTbyte index);

    cuConstEventLine GetConstPtr(CTbyte index);

    void ScanEventTypes(CTuint eventCount);

    void ScanEvents(CTuint eventCount);

    void CompactClippedEvents(CTuint eventCount);

    ~EventLine(void);
};

struct EventLines
{
    EventLine eventLines[3];
    CTbyte toggleIndex;

    EventLines(void) : toggleIndex(0)
    {

    }

    void Resize(CTuint size, cudaStream_t pStream)
    {
        if(eventLines[0].Size() >= size) return;
        for(int i = 0; i < 3; ++i)
        {
            eventLines[i].Resize((CTuint)(1.2 * size));
        }
        BindToConstantMemory(pStream);
    }

    void BindToConstantMemory(cudaStream_t pStream);

    void Toggle(void)
    {
        toggleIndex ^= 1;
        for(int i = 0; i < 3; ++i)
        {
            eventLines[i].toggleIndex = toggleIndex;
        }
    }
};

struct cuEventLineTriple
{
    cuEventLine lines[3];
};

struct cuConstEventLineTriple
{
    cuConstEventLine lines[3];
};

template <
    CTuint count, 
    typename T
>
struct Tuple
{
    T* ts[count];
};

template <
    CTuint count, 
    typename T
>
struct ConstTuple
{
    const T* __restrict ts[count];
};

struct dpClipMask
{
    nutty::DeviceBuffer<CTclipMask_t> mask0;
    nutty::DeviceBuffer<CTclipMask_t> mask1;
    nutty::DeviceBuffer<CTclipMask_t> mask2;

    nutty::DeviceBuffer<CTuint> scannedMask0;
    nutty::DeviceBuffer<CTuint> scannedMask1;
    nutty::DeviceBuffer<CTuint> scannedMask2;

    nutty::DeviceBuffer<CTuint> scannedSums0;
    nutty::DeviceBuffer<CTuint> scannedSums1;
    nutty::DeviceBuffer<CTuint> scannedSums2;

    nutty::DeviceBuffer<CTreal> newSplits0;
    nutty::DeviceBuffer<CTreal> newSplits1;
    nutty::DeviceBuffer<CTreal> newSplits2;

    nutty::DeviceBuffer<CTuint> index0;
    nutty::DeviceBuffer<CTuint> index1;
    nutty::DeviceBuffer<CTuint> index2;

    dpClipMask(void)
    {

    }

    ~dpClipMask(void)
    {

    }

    void Resize(size_t newSize, cudaStream_t pStream);
};

struct dpEventLine
{
    DoubleBuffer<IndexedEvent> indexedEvent;
    DoubleBuffer<CTeventType_t> type;
    DoubleBuffer<CTuint> primId;
    DoubleBuffer<BBox> ranges;

    nutty::DeviceBuffer<CTuint> typeStartScanned;
    nutty::DeviceBuffer<CTuint> scannedEventTypeEndMask;
    nutty::DeviceBuffer<CTuint> scannedEventTypeEndMaskSums;
    nutty::DeviceBuffer<CTeventMask_t> mask;

    CTbyte toggleIndex;

    dpEventLine(void)
    {

    }

    cuEventLine GetPtr(CTbyte index);

    ~dpEventLine(void)
    {

    }

    void Resize(size_t newSize);
};

struct dpEventLines
{
    dpEventLine eventLines[3];

    void Resize(size_t newSize, cudaStream_t pStream);
};

class cudpKDTree : public cuKDTree
{
private:
    void GrowLeafNodeMemory(CTuint size = (CTuint)-1);
    void GrowLeafContentMemory(CTuint size = (CTuint)-1);
    void GrowEventMemory(CTuint size = (CTuint)-1);
    void GrowNodeMemory(CTuint size = (CTuint)-1);
    void GrowPerLevelNodeMemory(CTuint size = (CTuint)-1);

    Split m_splits;
    nutty::DeviceBuffer<IndexedSAHSplit> m_splits_IndexedSplit;
    nutty::DeviceBuffer<CTreal> m_splits_Plane;
    nutty::DeviceBuffer<CTaxis_t> m_splits_Axis;
    nutty::DeviceBuffer<CTuint> m_splits_Above;
    nutty::DeviceBuffer<CTuint> m_splits_Below;

    DoubleBuffer<CTuint> m_eventNodeIndex;

    dpClipMask m_clipsMask;
    dpEventLines m_eventLines;

    nutty::DeviceBuffer<CTeventIsLeaf_t> m_eventIsLeaf;

    nutty::DeviceBuffer<CTuint> m_eventIsLeafScanned;
    nutty::DeviceBuffer<CTuint> m_eventIsLeafScannedSums;

    nutty::DeviceBuffer<CTuint> m_interiorCountScanned;
    nutty::DeviceBuffer<CTuint> m_leafContentScanned;
    nutty::DeviceBuffer<CTuint> m_scannedInteriorContent;

    nutty::DeviceBuffer<CTnodeIsLeaf_t> m_gotLeaves;

    nutty::DeviceBuffer<CTuint> m_maskedInteriorContent;
    nutty::DeviceBuffer<CTuint> m_maskedleafContent;

    nutty::DeviceBuffer<CTuint> m_lastNodeContentStartAdd;

    nutty::DeviceBuffer<CTuint> m_activeNodesThisLevel;
    nutty::DeviceBuffer<CTuint> m_activeNodes;
    nutty::DeviceBuffer<CTuint> m_newActiveNodes;
    nutty::DeviceBuffer<CTnodeIsLeaf_t> m_activeNodesIsLeaf;
    //nutty::DeviceBuffer<CTnodeIsLeaf_t> m_activeNodesIsLeafCompacted;
    nutty::DeviceBuffer<CTuint> m_leafCountScanned;

    DoubleBuffer<BBox> m_nodes_BBox;

    nutty::DeviceBuffer<CTuint> m_newNodesContentCount;
    nutty::DeviceBuffer<CTuint> m_newNodesContentStartAdd;

public:
    cudpKDTree(void) { }

    CT_RESULT Update(void);

    const void* GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const;

    add_uuid_header(cudpKDTree);
};

class cuKDTreeScan : public cuKDTree
{
private:
    nutty::cuStream m_compactStreams[5];
    void ClearBuffer(void);
    void GrowSplitMemory(CTuint size);
    void GrowNodeMemory(void);
    void GrowPerLevelNodeMemory(CTuint newSize);
    void InitBuffer(void);

    void PrintStatus(const char* msg = NULL);

    void ValidateTree(void);

    void ScanClipMaskTriples(CTuint eventCount);

    void ScanEventTypesTriples(CTuint eventCount);

    void ComputeSAH_Splits(
        CTuint nodeCount, 
        CTuint eventCount, 
        const CTuint* nodesContentCount);

    CTuint CheckRangeForLeavesAndPrepareBuffer(nutty::DeviceBuffer<CTnodeIsLeaf_t>::iterator& isLeafBegin, CTuint nodeOffset, CTuint range);

    MakeLeavesResult MakeLeaves(
        nutty::DeviceBuffer<CTnodeIsLeaf_t>::iterator& isLeafBegin,
        CTuint g_nodeOffset, 
        CTuint nodeOffset, 
        CTuint nodeCount, 
        CTuint eventCount, 
        CTuint currentLeafCount, 
        CTuint leafContentOffset,
        CTuint initNodeToLeafIndex,
        CTbyte gotLeaves = 1);

    nutty::HostBuffer<CTuint> m_hNodesContentCount;

    Split m_splits;
    nutty::DeviceBuffer<IndexedSAHSplit> m_splits_IndexedSplit;
    nutty::DeviceBuffer<CTreal> m_splits_Plane;
    nutty::DeviceBuffer<CTaxis_t> m_splits_Axis;
    nutty::DeviceBuffer<CTuint> m_splits_Above;
    nutty::DeviceBuffer<CTuint> m_splits_Below;

    DoubleBuffer<CTuint> m_eventNodeIndex;

    ClipMask m_clipsMask;

    //EventLine m_events3[3];
    EventLines m_eventLines;

    nutty::DeviceBuffer<CTeventIsLeaf_t> m_eventIsLeaf;

    nutty::DeviceBuffer<CTuint> m_eventIsLeafScanned;
    nutty::DeviceBuffer<CTuint> m_eventIsLeafScannedSums;

    nutty::Scanner m_eventIsLeafScanner;
    nutty::Scanner m_leafCountScanner;
    nutty::DeviceBuffer<CTuint> m_interiorCountScanned;
    nutty::Scanner m_interiorContentScanner;
    nutty::DeviceBuffer<CTuint> m_leafContentScanned;
    nutty::TScanner<CTuint3, ScanPrimitiveStruct<CTuint3>> m_typeScanner;
    nutty::DeviceBuffer<CTbyte3> m_types3;

    nutty::DeviceBuffer<CTnodeIsLeaf_t> m_gotLeaves;

    nutty::DeviceBuffer<CTuint> m_maskedInteriorContent;
    nutty::DeviceBuffer<CTuint> m_maskedleafContent;

    nutty::DeviceBuffer<CTuint> m_lastNodeContentStartAdd;

    nutty::DeviceBuffer<CTuint> m_activeNodesThisLevel;
    nutty::DeviceBuffer<CTuint> m_activeNodes;
    nutty::DeviceBuffer<CTuint> m_newActiveNodes;
    nutty::DeviceBuffer<CTnodeIsLeaf_t> m_activeNodesIsLeaf;
/*    nutty::DeviceBuffer<CTnodeIsLeaf_t> m_activeNodesIsLeafCompacted;*/

    DoubleBuffer<BBox> m_nodesBBox;

    nutty::DeviceBuffer<CTuint> m_newNodesContentCount;
    nutty::DeviceBuffer<CTuint> m_newNodesContentStartAdd;

    nutty::cuStreamPool m_pool;

    cuAsyncDtHCopy<CTuint> m_dthAsyncNodesContent;

    cuAsyncDtHCopy<CTuint> m_dthAsyncIntCopy;
    cuAsyncDtHCopy<CTnodeIsLeaf_t> m_dthAsyncByteCopy;

public:
    cuKDTreeScan(void)
    {
    }

    CT_RESULT Update(void);

    ~cuKDTreeScan(void)
    {

    }

    add_uuid_header(cuKDTreeScan);
};