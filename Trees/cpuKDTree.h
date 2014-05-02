#pragma once
#include "ct_runtime.h"

#include "geometry.h"
#include "memory.h"

enum SplitAxis
{
    eX = 0,
    eY = 1,
    eZ = 2
};

struct cpuTreeNode : public ICTTreeNode
{
    bool isLeaf;
    cpuTreeNode* left;
    cpuTreeNode* right;
    float split;
    SplitAxis splitAxis;
    std::vector<uint> geometry;
    AABB m_aabb;
    bool visited;

    uint leftAdd;
    uint rightAdd;

    cpuTreeNode(void) : left(NULL), right(NULL), isLeaf(false), visited(false), leftAdd(-1), rightAdd(-1)
    {
    }

    ~cpuTreeNode(void)
    {
        CT_SAFE_FREE(left);
        CT_SAFE_FREE(right);
    }
};

class cpuKDTree : public ICTTree
{
private:
    uint m_depth;
    cpuTreeNode* m_root;
    uint address;
    std::vector<ICTGeometry*> geometry;

    void _CreateTree(cpuTreeNode* parent, uint depth);

public:
    cpuKDTree(void) : m_root(NULL), m_depth(-1), address(0)
    {
    }

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo)
    {
        return CT_SUCCESS;
    }

    CT_RESULT Init(uint flags = 0);

    CT_RESULT Update(void);

    ICTTreeNode* GetNodesEntryPtr(void)
    {
        return m_root;
    }

    CT_RESULT AddGeometry(ICTGeometry* geo);

    void DebugDraw(ICTTreeDebugLayer* dbLayer);

    uint GetDepth(void)
    {
        return m_depth;
    }

    uint GetNodesCount(void);

    void SetDepth(byte depth)
    {
        m_depth = depth;
    }

    ~cpuKDTree(void);
};