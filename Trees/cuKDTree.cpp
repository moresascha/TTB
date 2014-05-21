#include "cuKDTree.h"
#include <Copy.h>
#include <HostPtr.h>
#include "ct.h"
#include "check_state.h"
#include "geometry.h"
#include "output.h"
#include "shared_kernel.h"

__forceinline CTuint GenerateDepth(CTuint N)
{
    return N > 32 ? (CTuint)(8.5 + 1.3 * log(N)) : 1; //PBR
}

cuKDTree::cuKDTree(void): m_depth(-1), m_initialized(false)
{
}

CT_RESULT cuKDTree::Init(CTuint flags)
{
    return CT_SUCCESS;
}

CT_RESULT cuKDTree::Update(void)
{
    m_depth = (byte)min(32, max(1, (m_depth == -1 ? GenerateDepth((CTuint)m_primitives.Size() / 3) : m_depth)));

    CTuint maxNodeCount = max(1, (1 << (uint)(m_depth)) - 1);
    m_nodesBBox.Resize(maxNodeCount);
    m_nodesIsLeaf.Resize(maxNodeCount);
    m_nodesSplit.Resize(maxNodeCount);
    m_nodesStartAdd.Resize(maxNodeCount);
    m_nodesSplitAxis.Resize(maxNodeCount);
    m_nodesContentCount.Resize(maxNodeCount);
    m_nodesAbove.Resize(maxNodeCount);
    m_nodesBelow.Resize(maxNodeCount);
    m_hNodesContentCount.Resize(elemsOnLevel(m_maxDepth-1));

    return CT_INVALID_OPERATION;
}

CT_RESULT cuKDTree::DebugDraw(ICTTreeDebugLayer* dbLayer) const
{

    return CT_SUCCESS;
}

CT_RESULT cuKDTree::AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle)
{

    return CT_OPERATION_NOT_SUPPORTED;
}

void cuKDTree::SetDepth(CTbyte depth)
{
    m_depth = clamp(depth, 0, m_maxDepth);
}

const CTGeometryHandle* cuKDTree::GetGeometry(CTuint* gc)
{
    return NULL;
}

const CTreal* cuKDTree::GetRawPrimitives(CTuint* bytes) const
{
    *bytes = m_primitives.Size() * sizeof(CTreal3);
    const CTreal3* ptr = m_primitives.Begin()();
    return (const CTreal*)ptr;
}

CT_RESULT cuKDTree::Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData /* = NULL */)
{
    return CT_OPERATION_NOT_SUPPORTED;
}

void cuKDTree::TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix)
{
    auto it = m_handleRangeMap.find(handle);
    if(it != m_handleRangeMap.end())
    {
        GeometryRange range = it->second;
        cudaTransformVector(m_primitives.Begin() + range.start, m_tprimitives.Begin() + range.start, matrix, range.end - range.start);
    }
}

const void* cuKDTree::GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const
{
    return NULL;
}

CT_RESULT cuKDTree::AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle)
{
    GeometryRange range;
    nutty::HostPtr<const float3> h_p = nutty::HostPtr_Cast<const float3>(memory);
    range.start = m_primitives.Size()/3;
    range.end = range.start + elements/3;
    m_primitives.PushBack(h_p, elements);
    *handle = m_handleRangeMap.size();
    m_handleRangeMap.insert(std::pair<CTGeometryHandle, GeometryRange>(*handle, range));
    return CT_SUCCESS;
}

cuKDTree::~cuKDTree(void)
{

}
