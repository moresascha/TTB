#pragma once
#include "tree.h"
#include <Nutty.h>
#include <DeviceBuffer.h>
#include "memory.h"

class cuKDTree : public ICTTree
{
private:
    ICTTreeNode* m_node;
    CT_GEOMETRY_TOPOLOGY m_topo;
    CTuint m_nodesCount;
    CTuint m_elementCount;
    byte m_depth;
    byte m_maxDepth;
    nutty::cuModule* m_pCudaModule;
    CTuint m_flags;
    CTbool m_initialized;

    void* m_devicePositionView;

public:
    cuKDTree(void);

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo);

    CT_RESULT Init(CTuint flags = 0);

    CT_RESULT AddGeometry(ICTGeometry* geo);

    void DebugDraw(ICTTreeDebugLayer* dbLayer)
    {

    }

    void OnPrimitiveMoved(const ICTPrimitive* prim)
    {

    }

    void OnGeometryMoved(const ICTGeometry* geo)
    {

    }

    CT_RESULT AllocateGeometry(ICTGeometry** geo)
    {
        return CT_SUCCESS;
    }

    CT_RESULT Traverse(ITraverse* traverse)
    {

    }

    CT_RESULT Update(void);

    ICTTreeNode* GetRoot(void);

    uint GetDepth(void);

    uint GetNodesCount(void);

    CT_RESULT QueryInterface(CTuuid id, void** ppInterface);

    uint GetInteriorNodesCount(void) const
    {
        return 0;
    }

    uint GetLeafNodesCount(void) const
    {
        return 0;
    }
    
    void SetDepth(byte d)
    {

    }

    const ICTGeometry* const* GetGeometry(CTuint* gc) const
    {
        return NULL;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return CT_TRIANGLES;
    }

    CT_TREE_DEVICE GetDeviceType(void) const
    {
        return eCT_GPU;
    }

    ~cuKDTree(void);

    add_uuid_header(cuKDTree);
};