#include "cuKDTree.h"
#include <Copy.h>
#include <HostPtr.h>
#include "ct.h"
#include "check_state.h"
#include "geometry.h"
#include "output.h"
#include "shared_kernel.h"
#include "ct_debug.h"
#include "vec_functions.h"

cuKDTree::cuKDTree(void): m_depth(0xFF), m_initialized(false)
{
    m_pStream = m_stream();
}

CT_RESULT cuKDTree::Init(CTuint flags)
{
    return CT_SUCCESS;
}

void cuKDTree::_DebugDrawNodes(CTuint parent, AABB aabb, ICTTreeDebugLayer* dbLayer) const
{
    if(m_nodes_IsLeaf[parent])
    {
        return;
    }

    CT_SPLIT_AXIS axis = (CT_SPLIT_AXIS)m_nodes_SplitAxis[parent];
    CTreal split = m_nodes_Split[parent];

    CTreal3 start;
    CTreal3 end;

    setAxis(start, axis, split);
    setAxis(end, axis, split);

    CTint other0 = (axis + 1) % 3;
    CTint other1 = (axis + 2) % 3;

    setAxis(start, other0, getAxis(aabb.GetMin(), other0));
    setAxis(start, other1, getAxis(aabb.GetMin(), other1));
    setAxis(end, other0, getAxis(aabb.GetMax(), other0));
    setAxis(end, other1, getAxis(aabb.GetMin(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(aabb.GetMin(), other0));
    setAxis(start, other1, getAxis(aabb.GetMin(), other1));
    setAxis(end, other0, getAxis(aabb.GetMin(), other0));
    setAxis(end, other1, getAxis(aabb.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(aabb.GetMin(), other0));
    setAxis(start, other1, getAxis(aabb.GetMax(), other1));
    setAxis(end, other0, getAxis(aabb.GetMax(), other0));
    setAxis(end, other1, getAxis(aabb.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(aabb.GetMax(), other0));
    setAxis(start, other1, getAxis(aabb.GetMin(), other1));
    setAxis(end, other0, getAxis(aabb.GetMax(), other0));
    setAxis(end, other1, getAxis(aabb.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    CTuint left = m_nodes_LeftChild[parent];
    CTuint right = m_nodes_RightChild[parent];
    AABB l;
    AABB r;
    splitAABB(&aabb, split, axis, &l, &r);
    _DebugDrawNodes(left, l, dbLayer);
    _DebugDrawNodes(right, r, dbLayer);
}

CT_RESULT cuKDTree::DebugDraw(ICTTreeDebugLayer* dbLayer) const
{
    checkState(m_initialized);
    if(!m_sceneBBox.Size())
    {
        return CT_SUCCESS;
    }
    AABB aabb;
    aabb.m_min = m_sceneBBox[0].GetMin();
    aabb.m_max = m_sceneBBox[0].GetMax();
    dbLayer->DrawWiredBox(aabb);
    _DebugDrawNodes(0, aabb, dbLayer);
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
    *bytes = m_currentTransformedVertices.Bytes();
    const CTreal3* ptr = m_currentTransformedVertices.Begin()();
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
        cudaTransformVector(m_orginalVertices.Begin() + range.start, m_currentTransformedVertices.Begin() + range.start, matrix, range.end - range.start, m_pStream);
    }
}

const void* cuKDTree::GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const
{
    switch(type)
    {
    case eCT_LEAF_NODE_PRIM_IDS :
        {
            *byteCount = m_leafNodesContent.Bytes();
            return m_leafNodesContent.Begin()();
        } break;
    case eCT_LEAF_NODE_PRIM_START_INDEX :
        {
            *byteCount = m_leafNodesContentStart.Bytes();
            return m_leafNodesContentStart.Begin()();
        } break;
    case eCT_LEAF_NODE_PRIM_COUNT :
        {
            *byteCount = m_leafNodesContentCount.Bytes();
            return m_leafNodesContentCount.Begin()();
        } break;
    case eCT_NODE_PRIM_IDS :
        {
            *byteCount = 0;
            return NULL;
        } break;
    case eCT_PRIMITVES :
        {
            return GetRawPrimitives(byteCount);
        }
    case eCT_NODE_SPLITS :
        {
            *byteCount = m_nodes_Split.Bytes();
            return m_nodes_Split.Begin()();
        }
    case eCT_NODE_SPLIT_AXIS :
        {
            *byteCount = m_nodes_SplitAxis.Bytes();
            return m_nodes_SplitAxis.Begin()();
        }
    case eCT_NODE_RIGHT_CHILD :
        {
            *byteCount = m_nodes_RightChild.Bytes();
            return m_nodes_RightChild.Begin()();
        }
    case eCT_NODE_LEFT_CHILD :
        {
            *byteCount = m_nodes_LeftChild.Bytes();
            return m_nodes_LeftChild.Begin()();
        }
    case eCT_NODE_IS_LEAF :
        {
            *byteCount = m_nodes_IsLeaf.Bytes();
            return m_nodes_IsLeaf.Begin()();
        }
    case eCT_NODE_PRIM_COUNT :
        {
            *byteCount = 0;
            return NULL;
        }
    case eCT_NODE_PRIM_START_INDEX :
        {
            *byteCount = 0;
            return NULL;
        }
    case eCT_NODE_TO_LEAF_INDEX :
        {
            *byteCount = m_nodes_NodeIdToLeafIndex.Bytes();
            return m_nodes_NodeIdToLeafIndex.Begin()();
        }
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
