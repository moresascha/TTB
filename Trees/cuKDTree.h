#pragma once
#include "ct_tree.h"
#include <Nutty.h>
#include <DeviceBuffer.h>
#include "ct_memory.h"

class CudaMemoryView : public ICTMemoryView
{
private:
    nutty::DeviceBuffer<ctfloat3> m_cuPositions;

public:
    void* GetMemory(void)
    {
        return (void*)m_cuPositions.Begin()();
    }
};

class cuKDTree : public ICTTree
{
private:
    ICTTreeNode* m_node;
    CT_GEOMETRY_TOPOLOGY m_topo;
    uint m_nodesCount;
    uint m_elementCount;
    byte m_depth;
    byte m_maxDepth;
    nutty::cuModule* m_pCudaModule;
    uint m_flags;
    bool m_initialized;

    CudaMemoryView m_devicePositionView;

public:
    cuKDTree(void);

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo);

    CT_RESULT Init(uint flags = 0);

    CT_RESULT AddGeometry(ICTGeometry* geo);

    void DebugDraw(ICTTreeDebugLayer* dbLayer)
    {

    }

    CT_RESULT Update(void);

    ICTTreeNode* GetNodesEntryPtr(void);

    uint GetDepth(void);

    uint GetNodesCount(void);

    CT_RESULT QueryInterface(ct_uuid id, void** ppInterface);
    
    void SetDepth(byte d)
    {

    }

    ~cuKDTree(void);
};