#include "cuKDTree.h"
#include "ct.h"
#include "check_state.h"
#include "ct_geometry.h"

#include "ct_output.h"

cuKDTree::cuKDTree(void) : m_initialized(false)
{

}

CT_RESULT cuKDTree::SetTopology(CT_GEOMETRY_TOPOLOGY topo)
{
    checkState(m_initialized == false);

    m_topo = topo;
    return CT_SUCCESS;
}

CT_RESULT cuKDTree::Init(uint flags)
{
    checkState(m_initialized == false);

    m_flags = flags;

    m_initialized = true;

    return CT_SUCCESS;
}

CT_RESULT cuKDTree::Update(void)
{
    checkState(m_initialized);

    return CT_SUCCESS;
}

ICTTreeNode* cuKDTree::GetNodesEntryPtr(void)
{
    return m_node;
}

uint cuKDTree::GetDepth(void)
{
    return m_depth;
}

uint cuKDTree::GetNodesCount(void)
{
    return m_nodesCount;
}

CT_RESULT cuKDTree::AddGeometry(ICTGeometry* geo)
{
    checkState(geo->GetTopology() == m_topo);

    return CT_SUCCESS;
}

CT_RESULT cuKDTree::QueryInterface(ct_uuid id, void** ppInterface)
{
    if(id == __uuidof(ICTMemoryView))
    {
        *ppInterface = &m_devicePositionView;
        return CT_SUCCESS;
    }
    return ICTInterface::QueryInterface(id, ppInterface);
}

cuKDTree::~cuKDTree(void)
{

}
