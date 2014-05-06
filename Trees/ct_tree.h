#pragma once
#include "ct.h"

struct ICTTreeNode : public ICTInterface
{
    static ct_uuid uuid(void) { return "ICTTreeNode"; }

    ~ICTTreeNode(void) {}
};

class ICTTree : public ICTInterface
{
public:
    virtual CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo) = 0;

    virtual CT_RESULT Init(uint flags = 0) = 0;

    virtual CT_RESULT Update(void) = 0;

    virtual ICTTreeNode* GetRoot(void) = 0;

    virtual void DebugDraw(ICTTreeDebugLayer* dbLayer) = 0;

    virtual CT_RESULT AddGeometry(ICTGeometry* geo) = 0;

    virtual void SetDepth(byte depth) = 0;

    virtual uint GetDepth(void) = 0;

    virtual uint GetInteriorNodesCount(void) const = 0;

    virtual uint GetLeafNodesCount(void) const = 0;

    virtual CT_TREE_DEVICE GetDeviceType(void) const = 0;

    virtual CT_GEOMETRY_TOPOLOGY GetTopology(void) const = 0;

    virtual ~ICTTree(void) {}

    static ct_uuid uuid(void) { return "ICTTree"; }
};