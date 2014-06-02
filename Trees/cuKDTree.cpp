#include "cuKDTree.h"
#include <Copy.h>
#include <HostPtr.h>
#include "ct.h"
#include "check_state.h"
#include "geometry.h"
#include "output.h"
#include "shared_kernel.h"

cuKDTree::cuKDTree(void): m_depth(0xFF), m_initialized(false)
{
}

CT_RESULT cuKDTree::Init(CTuint flags)
{
    return CT_SUCCESS;
}

CT_RESULT cuKDTree::DebugDraw(ICTTreeDebugLayer* dbLayer) const
{

    return CT_SUCCESS;
}

CT_RESULT cuKDTree::AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle)
{
    if(geo->GetTopology() != CT_TRIANGLES)
    {
        return CT_INVALID_VALUE;
    }

    CTuint pc;
    const ICTPrimitive* const* prims = geo->GetPrimitives(&pc);
    std::vector<CTreal3> vertices;
    for(CTuint i = 0; i < pc; ++i)
    {
        const CTTriangle* tri = static_cast<const CTTriangle*>(prims[i]);
        for(byte j = 0; j < 3; ++j)
        {
            CTreal3 v;
            tri->GetValue(j, v);
            vertices.push_back(v);
        }
    }

    return AddGeometryFromLinearMemory((void*)&vertices[0], 3 * pc, handle);
}

void cuKDTree::SetDepth(CTbyte depth)
{
    //m_depth = clamp(depth, 0, m_depth);
}

const CTGeometryHandle* cuKDTree::GetGeometry(CTuint* gc)
{
    return NULL;
}

const CTreal* cuKDTree::GetRawPrimitives(CTuint* bytes) const
{
    *bytes = m_orginalVertices.ByteCount();
    const CTreal3* ptr = m_orginalVertices.Begin()();
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
        cudaTransformVector(m_orginalVertices.Begin() + range.start, m_currentTransformedVertices.Begin() + range.start, matrix, range.end - range.start);
    }
}

const void* cuKDTree::GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const
{
    switch(type)
    {
//     case eCT_LEAF_NODE_PRIM_IDS :
//         {
//             *byteCount = m_linearPerLeafNodePrimitives.ByteCount();
//             return m_linearPerLeafNodePrimitives.Begin();
//         } break;
//     case eCT_LEAF_NODE_PRIM_START_INDEX :
//         {
//             *byteCount = m_linearPerLeafNodePrimStartIndex.ByteCount();
//             return m_linearPerLeafNodePrimStartIndex.Begin();
//         } break;
//     case eCT_LEAF_NODE_PRIM_COUNT :
//         {
//             *byteCount = m_linearPerLeafNodePrimCount.Bytes();
//             return m_linearPerLeafNodePrimCount.Begin();
//         } break;
//     case eCT_NODE_PRIM_IDS :
//         {
//             *byteCount = m_linearPerNodePrimitives.Bytes();
//             return m_linearPerNodePrimitives.Begin();
//         } break;
//     case eCT_PRIMITVES :
//         {
//             return GetRawPrimitives(byteCount);
//         }
//     case eCT_NODE_SPLITS :
//         {
//             *byteCount = m_linearNodeSplits.Bytes();
//             return m_linearNodeSplits.Begin();
//         }
//     case eCT_NODE_SPLIT_AXIS :
//         {
//             *byteCount = m_linearNodeSplitAxis.Bytes();
//             return m_linearNodeSplitAxis.Begin();
//         }
//     case eCT_NODE_RIGHT_CHILD :
//         {
//             *byteCount = m_linearNodeRight.Bytes();
//             return m_linearNodeRight.Begin();
//         }
//     case eCT_NODE_LEFT_CHILD :
//         {
//             *byteCount = m_linearNodeLeft.Bytes();
//             return m_linearNodeLeft.Begin();
//         }
//     case eCT_NODE_IS_LEAF :
//         {
//             *byteCount = m_linearNodeIsLeaf.Bytes();
//             return m_linearNodeIsLeaf.Begin();
//         }
//     case eCT_NODE_PRIM_COUNT :
//         {
//             *byteCount = m_linearNodePrimCount.Bytes();
//             return m_linearNodePrimCount.Begin();
//         }
//     case eCT_NODE_PRIM_START_INDEX :
//         {
//             *byteCount = m_linearNodePrimStartIndex.Bytes();
//             return m_linearNodePrimStartIndex.Begin();
//         }
//     case eCT_NODE_TO_LEAF_INDEX :
//         {
//             *byteCount = m_linearNodeToLeafIndex.Bytes();
//             return m_linearNodeToLeafIndex.Begin();
//         }
    default : *byteCount = 0; return NULL;
    }
}

CT_RESULT cuKDTree::AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle)
{
    CTulong h = (CTulong)m_linearGeoHandles.size();

    GeometryRange range;
    range.start = m_orginalVertices.Size();
    range.end = range.start + elements;

    nutty::HostPtr<const float3> h_p = nutty::HostPtr_Cast<const float3>(memory);    
    m_orginalVertices.PushBack(h_p, elements);
    m_currentTransformedVertices.PushBack(h_p, elements);

    m_handleRangeMap.insert(std::pair<CTulong, GeometryRange>(h, range));
    m_linearGeoHandles.push_back(h);
    *handle = (CTGeometryHandle)m_linearGeoHandles.back();

//     m_primAABBs.Resize(m_primAABBs.Size() + elements/3);
//     auto it = m_primAABBs.Begin() + range.start / 3;
// 
//     cudaCreateTriangleAABBs((m_orginalPrimitives.Begin() + range.start)(), it(), elements/3);

    return CT_SUCCESS;
}

cuKDTree::~cuKDTree(void)
{

}
