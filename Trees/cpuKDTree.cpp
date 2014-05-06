#include "cpuKDTree.h"
#include <algorithm>
#include "ct_output.h"
#include "ct_debug.h"

// struct ICTTreeNode : public ICTInterface
// {
//     bool isLeaf;
//     ICTTreeNode* left;
//     ICTTreeNode* right;
//     float split;
//     SplitAxis splitAxis;
//     std::vector<ICTGeometry*> geometry;
//     AABB m_aabb;
// 
//     ICTTreeNode(void) : left(NULL), right(NULL), isLeaf(false)
//     {
//     }
// 
//     ~ICTTreeNode(void)
//     {
//         CT_SAFE_FREE(left);
//         CT_SAFE_FREE(right);
//     }
// };

static int fls(int f)
{
    int order;
    for (order = 0; f != 0;f >>= 1, order++);
    return order;
}

static int ilog2(int f)
{
    return fls(f) - 1;
}

static enum SplitAxis getLongestAxis(float3 v)
{
    float m = fmaxf(v.x, fmaxf(v.y, v.z));
    return m == v.x ? eX : m == v.y ? eY : eZ; //v.x > v.y && v.x > v.z ? eX : v.y > v.x && v.y > v.z ? eY : eZ;
}

static float getAxis(const ctfloat3& vec, byte axis)
{
    switch(axis)
    {
    case 0 : return vec.x;
    case 1 : return vec.y;
    case 2 : return vec.z;
    }
    return 0;
}

static void setAxis(ctfloat3& vec, byte axis, float v)
{
    switch(axis)
    {
    case 0 : vec.x = v; return;
    case 1 : vec.y = v; return;
    case 2 : vec.z = v; return;
    }
}

float3 getAxisScale(const AABB& aabb)
{
    return make_float3(aabb.GetMax().x - aabb.GetMin().x, aabb.GetMax().y - aabb.GetMin().y, aabb.GetMax().z - aabb.GetMin().z);
}

float __device__ getArea(AABB& aabb)
{
    float3 axisScale = getAxisScale(aabb);
    return 2 * axisScale.x * axisScale.y + 2 * axisScale.x * axisScale.z + 2 * axisScale.y * axisScale.z;
}

float getSAH(AABB& node, int axis, float split, int primBelow, int primAbove, float traversalCost = 0.125f, float isectCost = 1)
{
    float cost = FLT_MAX;
    if(split > getAxis(node.GetMin(), axis) && split < getAxis(node.GetMax(), axis))
    {
        float3 axisScale = getAxisScale(node);
        float invTotalSA = 1.0f / getArea(node);
        int otherAxis0 = (axis+1) % 3;
        int otherAxis1 = (axis+2) % 3;
        float belowSA = 
            2 * 
            (getAxis(axisScale, otherAxis0) * getAxis(axisScale, otherAxis1) + 
            (split - getAxis(node.GetMin(), axis)) * 
            (getAxis(axisScale, otherAxis0) + getAxis(axisScale, otherAxis1)));

        float aboveSA = 
            2 * 
            (getAxis(axisScale, otherAxis0) * getAxis(axisScale, otherAxis1) + 
            (getAxis(node.GetMax(), axis) - split) * 
            (getAxis(axisScale, otherAxis0) + getAxis(axisScale, otherAxis1)));    

        float pbelow = belowSA * invTotalSA;
        float pabove = aboveSA * invTotalSA;
        float bonus = (primAbove == 0 || primBelow == 0) ? 1.0f : 0.0f;
        cost = traversalCost + isectCost * (1.0f - bonus) * (pbelow * primBelow + pabove * primAbove);
    }
    return cost;
}

enum EdgeType
{
    eEdgeStart,
    eEdgeEnd
};

struct EdgeEvent
{
    float split;
    EdgeType type;
    bool EdgeEvent::operator<(const EdgeEvent &o) const
    {
        return this->split < o.split;
    }
};

void cpuKDTree::_CreateTree(cpuTreeNode* parent, uint depth)
{
    //ct_printf("%d\n", address);
    m_address++;

    if(depth == m_depth || parent->primitives.size() < 4)
    {
        m_leafNodesCount++;
        parent->isLeaf = true;
        return;
    }

    m_interiorNodesCount++;

    std::vector<EdgeEvent> events;

    SplitAxis axis = getLongestAxis(parent->m_aabb.GetMax() - parent->m_aabb.GetMin());

    for(int i = 0; i < parent->primitives.size(); ++i)
    {

        const ICTAABB& geoAABB = primitives[parent->primitives[i]]->GetAABB();

        float start = getAxis(geoAABB.GetMin(), axis);
        float end = getAxis(geoAABB.GetMax(), axis);

        EdgeEvent startEvent;
        startEvent.split = start;
        startEvent.type = eEdgeStart;
        events.push_back(startEvent);

        EdgeEvent endEvent;
        endEvent.type = eEdgeEnd;
        endEvent.split = end;
        events.push_back(endEvent);
    }

    std::sort(events.begin(), events.end());

    float currentBestSAH = FLT_MAX;
    float currentSplit = 0;
    int primsBelow = 0;
    int primsAbove = (int)parent->primitives.size();

    for(int i = 0; i < events.size(); ++i)
    {
        EdgeEvent& event = events[i];
        
        if(event.type == eEdgeEnd)
        {
            primsAbove--;
        }

        float sah = getSAH(parent->m_aabb, axis, event.split, primsBelow, primsAbove);

        //ct_printf("sah=%f, axis=%d, split=%f\n", sah, axis, event.split);

        if(sah < currentBestSAH)
        {
            currentBestSAH = sah;
            currentSplit = event.split;
        }

        if(event.type == eEdgeStart)
        {
            primsBelow++;
        }
    }

    if(currentBestSAH == FLT_MAX)
    {
        parent->isLeaf = true;
        return;
    }

    //ct_printf("Found best: sah=%f, axis=%d, split=%f\n\n\n ---------------------- \n\n\n", currentBestSAH, axis, currentSplit);

    parent->split = currentSplit;
    parent->splitAxis = axis;

    parent->left = CTMemAllocObject<cpuTreeNode>();
    parent->right = CTMemAllocObject<cpuTreeNode>();

    parent->left->m_aabb = parent->m_aabb;
    parent->left->m_aabb.ShrinkMax(axis, currentSplit);
    parent->left->split = 0;
    parent->left->splitAxis = (SplitAxis)-1;
    parent->isLeaf = false;

    parent->right->m_aabb = parent->m_aabb;
    parent->right->m_aabb.ShrinkMin(axis, currentSplit);
    parent->right->split = 0;
    parent->right->splitAxis = (SplitAxis)-1;
    parent->isLeaf = false;

    for(uint i = 0; i < parent->primitives.size(); ++i)
    {
        const ICTPrimitive* prim = primitives[parent->primitives[i]];
        
        float mini = getAxis(prim->GetAABB().GetMin(), axis);
        float maxi = getAxis(prim->GetAABB().GetMax(), axis);

        if(mini < currentSplit)
        {
            parent->left->primitives.push_back(parent->primitives[i]);
        }

        if(maxi > currentSplit)
        {
            parent->right->primitives.push_back(parent->primitives[i]);
        }
    }
    parent->leftAdd = m_address;
    _CreateTree(parent->left, depth + 1);
    parent->rightAdd = m_address;
    _CreateTree(parent->right, depth + 1);
}

void _DebugDrawNodes(cpuTreeNode* parent, ICTTreeDebugLayer* dbLayer, std::vector<const ICTGeometry*> geometry)
{
    if(parent->isLeaf)
    {
//         dbLayer->SetDrawColor(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
//         for(int i = 0; i < parent->geometry.size(); ++i)
//         {
//             dbLayer->DrawBox(geometry[parent->geometry[i]]->GetAABB());
//         }
        return;
    }

    SplitAxis axis = parent->splitAxis;
    float split = parent->split;

    float3 start;
    float3 end;

    setAxis(start, axis, split);
    setAxis(end, axis, split);

    int other0 = (axis + 1) % 3;
    int other1 = (axis + 2) % 3;

    dbLayer->SetDrawColor(0,0.2f,0);

    setAxis(start, other0, getAxis(parent->m_aabb.GetMin(), other0));
    setAxis(start, other1, getAxis(parent->m_aabb.GetMin(), other1));
    setAxis(end, other0, getAxis(parent->m_aabb.GetMax(), other0));
    setAxis(end, other1, getAxis(parent->m_aabb.GetMin(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(parent->m_aabb.GetMin(), other0));
    setAxis(start, other1, getAxis(parent->m_aabb.GetMin(), other1));
    setAxis(end, other0, getAxis(parent->m_aabb.GetMin(), other0));
    setAxis(end, other1, getAxis(parent->m_aabb.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(parent->m_aabb.GetMin(), other0));
    setAxis(start, other1, getAxis(parent->m_aabb.GetMax(), other1));
    setAxis(end, other0, getAxis(parent->m_aabb.GetMax(), other0));
    setAxis(end, other1, getAxis(parent->m_aabb.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(parent->m_aabb.GetMax(), other0));
    setAxis(start, other1, getAxis(parent->m_aabb.GetMin(), other1));
    setAxis(end, other0, getAxis(parent->m_aabb.GetMax(), other0));
    setAxis(end, other1, getAxis(parent->m_aabb.GetMax(), other1));
    dbLayer->DrawLine(start, end);

     _DebugDrawNodes(parent->left, dbLayer, geometry);
     _DebugDrawNodes(parent->right, dbLayer, geometry);
}

void cpuKDTree::DebugDraw(ICTTreeDebugLayer* dbLayer)
{
    srand(0);
    for(int i = 0; i < geometry.size(); ++i)
    {
        //dbLayer->DrawBox(geometry[i]->GetAABB());
    }

    dbLayer->DrawWiredBox(m_root->m_aabb);

    _DebugDrawNodes(m_root, dbLayer, geometry);
}

uint GenerateDepth(uint N)
{
    return (uint)(8.5 + 1.3 * log(N));
}

void cpuKDTree::OnPrimitiveMoved(const ICTPrimitive* prim)
{
    uint vc;
    const ICTVertex* const* v = prim->GetVertices(&vc);
    for(uint j = 0; j < vc; ++j)
    {
        m_root->m_aabb.AddVertex(v[j]);
    }
}

void cpuKDTree::OnGeometryMoved(const ICTGeometry* geo)
{
    uint pc = 0;
    const ICTPrimitive* const* prims = geo->GetPrimitives(&pc);
    for(uint i = 0; i < pc; ++i)
    {
        OnPrimitiveMoved(prims[i]);
    }
}

uint cpuKDTree::GetInteriorNodesCount(void) const
{
    return m_interiorNodesCount;
}

uint cpuKDTree::GetLeafNodesCount(void) const
{
    return m_leafNodesCount;
}

CT_RESULT cpuKDTree::Update(void)
{
    if(m_root)
    {
        m_leafNodesCount = 0;
        m_interiorNodesCount = 0;
        m_address = 0;
        CT_SAFE_FREE(m_root->left);
        CT_SAFE_FREE(m_root->right);
        m_depth = min(128, max(1, (m_depth == -1 ? GenerateDepth((uint)geometry.size()) : m_depth)));
        _CreateTree(m_root, 0);
        return CT_SUCCESS;
    }

    return CT_INVALID_OPERATION;
}

CT_RESULT cpuKDTree::AddGeometry(ICTGeometry* geo)
{
    uint pc = 0;
    const ICTPrimitive* const* prims = geo->GetPrimitives(&pc);
    for(uint i = 0; i < pc; ++i)
    {
        uint vc;
        const ICTVertex* const* v = prims[i]->GetVertices(&vc);
        for(uint j = 0; j < vc; ++j)
        {
            m_root->m_aabb.AddVertex(v[j]);
        }
        m_root->primitives.push_back((uint)m_root->primitives.size());
        primitives.push_back(prims[i]);
    }
    geometry.push_back(geo);
    return CT_SUCCESS;
}

CT_RESULT cpuKDTree::Init(uint flags /* = 0 */)
{
    if(m_root)
    {
        return CT_INVALID_OPERATION;
    }

    m_root = CTMemAllocObject<cpuTreeNode>();

    return CT_SUCCESS;
}

cpuKDTree::~cpuKDTree(void)
{
    CT_SAFE_FREE(m_root);
}