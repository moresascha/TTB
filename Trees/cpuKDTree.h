#pragma once
#include "ct_runtime.h"
#include <vector>
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
    std::vector<uint> primitives;
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
    friend class Geometry;
private:
    uint m_depth;
    cpuTreeNode* m_root;
    uint m_address;
    uint m_interiorNodesCount;
    uint m_leafNodesCount;
    std::vector<const ICTGeometry*> geometry;
    std::vector<const ICTPrimitive*> primitives;

    void _CreateTree(cpuTreeNode* parent, uint depth);

protected:
    void OnPrimitiveMoved(const ICTPrimitive* prim);
    void OnGeometryMoved(const ICTGeometry* geo);

public:
    cpuKDTree(void) : m_root(NULL), m_depth(-1), m_address(0)
    {
    }

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo)
    {
        return CT_SUCCESS;
    }

    CT_RESULT Init(uint flags = 0);

    CT_RESULT Update(void);

    ICTTreeNode* GetRoot(void)
    {
        return m_root;
    }

    CT_RESULT AddGeometry(ICTGeometry* geo);

    void DebugDraw(ICTTreeDebugLayer* dbLayer);

    uint GetDepth(void)
    {
        return m_depth;
    }

    uint GetInteriorNodesCount(void) const;

    uint GetLeafNodesCount(void) const;

    void SetDepth(byte depth)
    {
        m_depth = depth;
    }

    CT_TREE_DEVICE GetDeviceType(void) const
    {
        return eCT_CPU;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return CT_TRIANGLES;
    }

    ~cpuKDTree(void);

    add_uuid_header(cpuKDTree);
};