#include "cpuKDTree.h"
#include <algorithm>
#include "ct_debug.h"
#include "geometry.h"
#include "memory.h"
#include <queue>

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

static enum CT_SPLIT_AXIS getLongestAxis(CTreal3 v)
{
    float m = fmaxf(v.x, fmaxf(v.y, v.z));
    return m == v.x ? eCT_X : m == v.y ? eCT_Y : eCT_Z;
}

static float getAxis(const CTreal3& vec, CTbyte axis)
{
    switch(axis)
    {
    case 0 : return vec.x;
    case 1 : return vec.y;
    case 2 : return vec.z;
    }
    return 0;
}

static void setAxis(CTreal3& vec, CTbyte axis, CTreal v)
{
    switch(axis)
    {
    case 0 : vec.x = v; return;
    case 1 : vec.y = v; return;
    case 2 : vec.z = v; return;
    }
}

float3 getAxisScale(const ICTAABB& aabb)
{
    return make_float3(aabb.GetMax().x - aabb.GetMin().x, aabb.GetMax().y - aabb.GetMin().y, aabb.GetMax().z - aabb.GetMin().z);
}

float __device__ getArea(const ICTAABB& aabb)
{
    float3 axisScale = getAxisScale(aabb);
    return 2 * axisScale.x * axisScale.y + 2 * axisScale.x * axisScale.z + 2 * axisScale.y * axisScale.z;
}

CTreal getSAH(const ICTAABB& node, CTint axis, CTreal split, CTint primBelow, CTint primAbove, CTreal traversalCost = 0.125f, CTreal isectCost = 1)
{
    CTreal cost = FLT_MAX;
    if(split > getAxis(node.GetMin(), axis) && split < getAxis(node.GetMax(), axis))
    {
        CTreal3 axisScale = getAxisScale(node);
        CTreal invTotalSA = 1.0f / getArea(node);
        CTint otherAxis0 = (axis+1) % 3;
        CTint otherAxis1 = (axis+2) % 3;
        CTreal belowSA = 
            2 * 
            (getAxis(axisScale, otherAxis0) * getAxis(axisScale, otherAxis1) + 
            (split - getAxis(node.GetMin(), axis)) * 
            (getAxis(axisScale, otherAxis0) + getAxis(axisScale, otherAxis1)));

        CTreal aboveSA = 
            2 * 
            (getAxis(axisScale, otherAxis0) * getAxis(axisScale, otherAxis1) + 
            (getAxis(node.GetMax(), axis) - split) * 
            (getAxis(axisScale, otherAxis0) + getAxis(axisScale, otherAxis1)));    

        CTreal pbelow = belowSA * invTotalSA;
        CTreal pabove = aboveSA * invTotalSA;
        CTreal bonus = (primAbove == 0 || primBelow == 0) ? 1.0f : 0.0f;
        cost = traversalCost + isectCost * (1.0f - bonus) * (pbelow * primBelow + pabove * primAbove);
    }
    return cost;
}

cpuKDTree* g_tree;

CTuint cpuTreeNode::GetPrimitive(CTuint index)
{
    return g_tree->GetPrimitive(primStartIndex + index);
}

cpuTreeNode::~cpuTreeNode(void)
{
}

// void cpuKDTree::_CreateTree(cpuTreeNode* parent, CTuint depth)
// {
//     //ct_printf("%d\n", address);
//     m_address++;
// 
//     for(int i = 0; i < depth; ++i)
//     {
//         ct_printf(" ");
//     }
//     parent->Print();
// 
//     if(depth == m_depth || parent->GetPrimitiveCount() < 4)
//     {
//         m_leafNodesCount++;
//         parent->isLeaf = true;
//         return;
//     }
// 
//     m_interiorNodesCount++;
// 
//     std::vector<EdgeEvent> events;
// 
//     CT_SPLIT_AXIS axis = getLongestAxis(m_linearNodeAABBs[parent->aabb].GetMax() - m_linearNodeAABBs[parent->aabb].GetMin());
//     
//     AABB* aabb;
// 
//     for(int i = 0; i < parent->GetPrimitiveCount(); ++i)
//     {
//         //m_linearPrimitiveMemory[parent->primitives[i]].GetAxisAlignedBB(geoAABB);
//         aabb = &m_linearPrimAABBs[parent->GetPrimitive(i)];
// 
//         float start = getAxis(aabb->GetMin(), axis);
//         float end = getAxis(aabb->GetMax(), axis);
// 
//         EdgeEvent startEvent;
//         startEvent.split = start;
//         startEvent.type = eEdgeStart;
//         events.push_back(startEvent);
// 
//         EdgeEvent endEvent;
//         endEvent.type = eEdgeEnd;
//         endEvent.split = end;
//         events.push_back(endEvent);
//     }
// 
//     std::sort(events.begin(), events.end());
// 
//     float currentBestSAH = FLT_MAX;
//     float currentSplit = 0;
//     int primsBelow = 0;
//     int primsAbove = (int)parent->GetPrimitiveCount();
// 
//     for(int i = 0; i < events.size(); ++i)
//     {
//         EdgeEvent& event = events[i];
//         
//         if(event.type == eEdgeEnd)
//         {
//             primsAbove--;
//         }
// 
//         float sah = getSAH(m_linearNodeAABBs[parent->aabb], axis, event.split, primsBelow, primsAbove);
// 
//         //ct_printf("sah=%f, axis=%d, split=%f\n", sah, axis, event.split);
// 
//         if(sah < currentBestSAH)
//         {
//             currentBestSAH = sah;
//             currentSplit = event.split;
//         }
// 
//         if(event.type == eEdgeStart)
//         {
//             primsBelow++;
//         }
//     }
// 
//     if(currentBestSAH == FLT_MAX)
//     {
//         parent->isLeaf = true;
//         return;
//     }
// 
//     //ct_printf("Found best: sah=%f, axis=%d, split=%f\n\n\n ---------------------- \n\n\n", currentBestSAH, axis, currentSplit);
// 
//     parent->split = currentSplit;
//     parent->splitAxis = axis;
// 
//     //left
//     parent->left = m_linearNodeMemory.Next();
//     parent->relLeft = parent->left - parent->me;
// 
//     cpuTreeNode& leftNode = m_linearNodeMemory[parent->left];
//     leftNode.aabb = m_linearNodeAABBs.Next();
//     leftNode.split = 0;
//     leftNode.splitAxis = (CT_SPLIT_AXIS)-1;
//     leftNode.isLeaf = false;
//     leftNode.me = parent->left;
// 
//     m_linearNodeAABBs[parent->left] = m_linearNodeAABBs[parent->aabb];
//     m_linearNodeAABBs[parent->left].ShrinkMax(axis, currentSplit);
// 
//     //right
//     parent->right = m_linearNodeMemory.Next();
//     parent->relRight = parent->right - parent->me;
// 
//     cpuTreeNode& rightNode = m_linearNodeMemory[parent->right];
//     rightNode.aabb = m_linearNodeAABBs.Next();
//     rightNode.split = 0;
//     rightNode.splitAxis = (CT_SPLIT_AXIS)-1;
//     rightNode.isLeaf = false;
//     rightNode.me = parent->right;
// 
//     m_linearNodeAABBs[parent->right] = m_linearNodeAABBs[parent->aabb];
//     m_linearNodeAABBs[parent->right].ShrinkMin(axis, currentSplit);
// 
//     parent->isLeaf = false;
// 
//     CTuint start = m_linearPerNodePrimitives.Size();
// 
//     leftNode.primStartIndex = start;
//     leftNode.primCount = 0;
// 
//     rightNode.primStartIndex = start;
//     rightNode.primCount = 0;
// 
//     m_linearSplitLeft.Reset();
//     m_linearSplitRight.Reset();
// 
//     for(CTuint i = 0; i < parent->GetPrimitiveCount(); ++i)
//     {
//         aabb = &m_linearPrimAABBs[parent->GetPrimitive(i)];
//         
//         float mini = getAxis(aabb->GetMin(), axis);
//         float maxi = getAxis(aabb->GetMax(), axis);
// 
//         if(mini < currentSplit)
//         {
//             leftNode.primCount++;
//             CTuint id = m_linearSplitLeft.Next();
//             m_linearSplitLeft[id] = parent->GetPrimitive(i);
//             //m_linearNodeMemory[parent->left].primitives.push_back(parent->primitives[i]);
//         }
// 
//         if(maxi > currentSplit)
//         {
//             rightNode.primCount++;
//             CTuint id = m_linearSplitRight.Next();
//             m_linearSplitRight[id] = parent->GetPrimitive(i);
//             //m_linearNodeMemory[parent->right].primitives.push_back(parent->primitives[i]);
//         }
//     }
// 
//     for(CTuint i = 0; i < m_linearSplitLeft.Size(); ++i)
//     {
//         CTuint id = m_linearPerNodePrimitives.Next();
//         m_linearPerNodePrimitives[id] = m_linearSplitLeft[i];
//     }
// 
//     for(CTuint i = 0; i < m_linearSplitRight.Size(); ++i)
//     {
//         CTuint id = m_linearPerNodePrimitives.Next();
//         m_linearPerNodePrimitives[id] = m_linearSplitRight[i];
//     }
// 
//     rightNode.primStartIndex += leftNode.primCount;
//     parent->leftAdd = m_address;
//     _CreateTree(&leftNode, depth + 1);
//     parent->rightAdd = m_address;
//     _CreateTree(&rightNode, depth + 1);
// }

void cpuKDTree::_CreateTree(void)
{
    std::queue<CTuint> queue;
    queue.push(m_root);

    m_events.reserve(m_linearPrimitiveMemory.size() * 2);

    while(!queue.empty())
    {
        CTuint nodeId = queue.front();
        cpuTreeNode* parent = &m_linearNodeMemory[nodeId];
        queue.pop();

        if(parent->depth == m_depth || parent->GetPrimitiveCount() < 4)
        {
            m_leafNodesCount++;
            parent->isLeaf = true;
            continue;
        }

        m_interiorNodesCount++;
        AABB& parentAABB = m_linearNodeAABBs[parent->aabb];

        CT_SPLIT_AXIS axis = getLongestAxis(parentAABB.GetMax() - parentAABB.GetMin());
        m_events.clear();
        for(int i = 0; i < parent->GetPrimitiveCount(); ++i)
        {
            AABB& aabb = m_linearPrimAABBs[parent->GetPrimitive(i)];

            float start = getAxis(aabb.GetMin(), axis);
            float end = getAxis(aabb.GetMax(), axis);

            EdgeEvent startEvent;
            startEvent.split = start;
            startEvent.type = eEdgeStart;
            m_events.push_back(startEvent);

            EdgeEvent endEvent;
            endEvent.type = eEdgeEnd;
            endEvent.split = end;
            m_events.push_back(endEvent);
        }

        std::sort(m_events.begin(), m_events.end());
                        
        float currentBestSAH = FLT_MAX;
        float currentSplit = 0;
        int primsBelow = 0;
        int primsAbove = (int)parent->GetPrimitiveCount();

        for(int i = 0; i < m_events.size(); ++i)
        {
            EdgeEvent& event = m_events[i];

            if(event.type == eEdgeEnd)
            {
                primsAbove--;
            }

            float sah = getSAH(parentAABB, axis, event.split, primsBelow, primsAbove);

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
            m_leafNodesCount++;
            continue;
        }

        //ct_printf("Found best: sah=%f, axis=%d, split=%f\n\n\n ---------------------- \n\n\n", currentBestSAH, axis, currentSplit);

        parent->split = currentSplit;
        parent->splitAxis = axis;

        //left
        CTuint l = m_linearNodeMemory.Next();
        CTuint r = m_linearNodeMemory.Next();   

        parent = &m_linearNodeMemory[nodeId];

        cpuTreeNode& leftNode = m_linearNodeMemory[l];
        cpuTreeNode& rightNode = m_linearNodeMemory[r]; 

        parent->left = l;
        parent->right = r; 

        parent->relLeft = parent->left - parent->me;
        parent->relRight = parent->right - parent->me;

        leftNode.aabb = m_linearNodeAABBs.Next();
        leftNode.split = 0;
        leftNode.splitAxis = (CT_SPLIT_AXIS)-1;
        leftNode.isLeaf = false;
        leftNode.me = parent->left;

        m_linearNodeAABBs[parent->left] = m_linearNodeAABBs[parent->aabb];
        m_linearNodeAABBs[parent->left].ShrinkMax(axis, currentSplit);

        rightNode.aabb = m_linearNodeAABBs.Next();
        rightNode.split = 0;
        rightNode.splitAxis = (CT_SPLIT_AXIS)-1;
        rightNode.isLeaf = false;
        rightNode.me = parent->right;

        m_linearNodeAABBs[parent->right] = m_linearNodeAABBs[parent->aabb];
        m_linearNodeAABBs[parent->right].ShrinkMin(axis, currentSplit);

        parent->isLeaf = false;

        CTuint start = m_linearPerNodePrimitives.Size();

        leftNode.primStartIndex = start;
        leftNode.primCount = 0;

        rightNode.primStartIndex = start;
        rightNode.primCount = 0;

        m_linearSplitLeft.Reset();
        m_linearSplitRight.Reset();

        for(CTuint i = 0; i < parent->GetPrimitiveCount(); ++i)
        {
            AABB& aabb = m_linearPrimAABBs[parent->GetPrimitive(i)];

            float mini = getAxis(aabb.GetMin(), axis);
            float maxi = getAxis(aabb.GetMax(), axis);

            if(mini < currentSplit)
            {
                leftNode.primCount++;
                CTuint id = m_linearSplitLeft.Next();
                m_linearSplitLeft[id] = parent->GetPrimitive(i);
            }

            if(maxi > currentSplit)
            {
                rightNode.primCount++;
                CTuint id = m_linearSplitRight.Next();
                m_linearSplitRight[id] = parent->GetPrimitive(i);
            }
        }

        for(CTuint i = 0; i < m_linearSplitLeft.Size(); ++i)
        {
            CTuint id = m_linearPerNodePrimitives.Next();
            m_linearPerNodePrimitives[id] = m_linearSplitLeft[i];
        }

        for(CTuint i = 0; i < m_linearSplitRight.Size(); ++i)
        {
            CTuint id = m_linearPerNodePrimitives.Next();
            m_linearPerNodePrimitives[id] = m_linearSplitRight[i];
        }

        rightNode.primStartIndex += leftNode.primCount;

        leftNode.depth = parent->depth + 1;
        rightNode.depth = parent->depth + 1;

        queue.push(parent->left);
        queue.push(parent->right);
    }
}

CT_RESULT cpuKDTree::Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData)
{
    switch(order)
    {
    case eCT_BREADTH_FIRST : 
        {
            for(CTuint i = 0; i < m_linearNodeMemory.Size(); ++i)
            {
                cb(&m_linearNodeMemory[i], userData);
            }
            return CT_SUCCESS;
        } break;
    default : return CT_INVALID_ENUM;
    }
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

    CT_SPLIT_AXIS axis = parent->splitAxis;
    CTreal split = parent->split;

    CTreal3 start;
    CTreal3 end;

    setAxis(start, axis, split);
    setAxis(end, axis, split);

    CTint other0 = (axis + 1) % 3;
    CTint other1 = (axis + 2) % 3;

    dbLayer->SetDrawColor(0,0.2f,0);

//     setAxis(start, other0, getAxis(parent->aabb->GetMin(), other0));
//     setAxis(start, other1, getAxis(parent->aabb->GetMin(), other1));
//     setAxis(end, other0, getAxis(parent->aabb->GetMax(), other0));
//     setAxis(end, other1, getAxis(parent->aabb->GetMin(), other1));
//     dbLayer->DrawLine(start, end);
// 
//     setAxis(start, other0, getAxis(parent->aabb->GetMin(), other0));
//     setAxis(start, other1, getAxis(parent->aabb->GetMin(), other1));
//     setAxis(end, other0, getAxis(parent->aabb->GetMin(), other0));
//     setAxis(end, other1, getAxis(parent->aabb->GetMax(), other1));
//     dbLayer->DrawLine(start, end);
// 
//     setAxis(start, other0, getAxis(parent->aabb->GetMin(), other0));
//     setAxis(start, other1, getAxis(parent->aabb->GetMax(), other1));
//     setAxis(end, other0, getAxis(parent->aabb->GetMax(), other0));
//     setAxis(end, other1, getAxis(parent->aabb->GetMax(), other1));
//     dbLayer->DrawLine(start, end);
// 
//     setAxis(start, other0, getAxis(parent->aabb->GetMax(), other0));
//     setAxis(start, other1, getAxis(parent->aabb->GetMin(), other1));
//     setAxis(end, other0, getAxis(parent->aabb->GetMax(), other0));
//     setAxis(end, other1, getAxis(parent->aabb->GetMax(), other1));
//     dbLayer->DrawLine(start, end);
// 
//      _DebugDrawNodes(parent->left, dbLayer, geometry);
//      _DebugDrawNodes(parent->right, dbLayer, geometry);
}

void cpuKDTree::DebugDraw(ICTTreeDebugLayer* dbLayer)
{
    srand(0);
//     for(CTint i = 0; i < geometry.size(); ++i)
//     {
//         //dbLayer->DrawBox(geometry[i]->GetAABB());
//     }

    //dbLayer->DrawWiredBox(*m_root->aabb);

    //_DebugDrawNodes(m_root, dbLayer, geometry);
}

CTuint GenerateDepth(CTuint N)
{
    return (CTuint)(8.5 + 1.3 * log(N));
}

void cpuKDTree::OnGeometryMoved(const CTGeometryHandle geo)
{

}

uint cpuKDTree::GetInteriorNodesCount(void) const
{
    return m_interiorNodesCount;
}

uint cpuKDTree::GetLeafNodesCount(void) const
{
    return m_leafNodesCount;
}

CTuint cpuKDTree::GetPrimitive(CTuint index)
{
    return m_linearPerNodePrimitives[index];
}

CT_RESULT cpuKDTree::Update(void)
{
    m_leafNodesCount = 0;
    m_interiorNodesCount = 0;
    m_address = 0;
    m_linearNodeMemory.Reset();
    m_linearNodeAABBs.Reset();
    g_tree = this;

    m_root = m_linearNodeMemory.Next();
    
    cpuTreeNode& root = m_linearNodeMemory[m_root];
    root.aabb = m_linearNodeAABBs.Next();
    m_linearNodeAABBs[root.aabb] = m_sceneAABB;
    root.primCount = m_linearPrimitiveMemory.size()/3;
    root.primStartIndex = 0;
    root.me = 0;
    root.depth = 0;
    m_linearPerNodePrimitives.Reset();
    for(CTuint i = 0; i < root.primCount; ++i)
    {
        m_linearPerNodePrimitives[m_linearPerNodePrimitives.Next()] = i;
    }

    m_depth = min(64, max(1, (m_depth == -1 ? GenerateDepth((uint)m_linearPrimitiveMemory.size()) : m_depth)));
    //_CreateTree(&root, 0);
    _CreateTree();
    return CT_SUCCESS;
}

CT_RESULT cpuKDTree::AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle)
{
    switch(geo->GetTopology())
    {
    case CT_TRIANGLES :
        {
            CTuint pc;
            const ICTPrimitive* const* prims = geo->GetPrimitives(&pc);
            CTuint startindex = (CTuint)m_linearPrimitiveMemory.size();
            for(CTuint i = 0; i < pc; ++i)
            {
                const CTTriangle* tri = static_cast<const CTTriangle*>(prims[i]);
                CTuint primAABB = m_linearPrimAABBs.Next();
                for(CTuint j = 0; j < 3; ++j)
                {
                    CTreal3 v;
                    tri->GetValue(j, v);
                    m_linearPrimitiveMemory.push_back(v);
                    m_linearPrimAABBs[primAABB].AddVertex(v);
                    m_sceneAABB.AddVertex(v);
                }
            }

            CTulong h = (CTulong)m_linearGeoHandles.size();
            
            GeometryRange range;
            range.start = startindex;
            range.end = startindex + pc;

            m_handleRangeMap.insert(std::pair<CTulong, GeometryRange>(h, range));
            m_linearGeoHandles.push_back(h);
            *handle = (CTGeometryHandle)m_linearGeoHandles.back();

            return CT_SUCCESS;
        } break;
    default : return CT_INVALID_ENUM;
    }
}

CTreal3 transform3f(const CTreal4* m3x3l, const CTreal3& vector)
{
    CTreal3 res;
    res.x = dot(make_float3(m3x3l[0].x, m3x3l[0].y, m3x3l[0].z), vector);
    res.y = dot(make_float3(m3x3l[1].x, m3x3l[1].y, m3x3l[1].z), vector);
    res.z = dot(make_float3(m3x3l[2].x, m3x3l[2].y, m3x3l[2].z), vector);
    return res;
}

void cpuKDTree::TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix)
{
    auto range = m_handleRangeMap.find(handle);
    if(range != m_handleRangeMap.end())
    {
         for(CTuint i = range->second.start; i < range->second.end; ++i)
         {
             CTreal d = 0.01;
             CTreal3& a = m_linearPrimitiveMemory[3 * i + 0];
             CTreal3& b = m_linearPrimitiveMemory[3 * i + 1];
             CTreal3& c = m_linearPrimitiveMemory[3 * i + 2];

             a = transform3f(matrix, a);
             b = transform3f(matrix, b);
             c = transform3f(matrix, c);

             m_linearPrimAABBs[i].Reset();
             m_linearPrimAABBs[i].AddVertex(a);
             m_linearPrimAABBs[i].AddVertex(b);
             m_linearPrimAABBs[i].AddVertex(c);

             m_sceneAABB.AddVertex(a);
             m_sceneAABB.AddVertex(b);
             m_sceneAABB.AddVertex(c);
         }
    }
}

CT_RESULT cpuKDTree::Init(CTuint flags /* = 0 */)
{
    return CT_SUCCESS;
}

cpuKDTree::~cpuKDTree(void)
{

}