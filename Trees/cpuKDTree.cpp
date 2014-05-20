#include "cpuKDTree.h"
#include <algorithm>
#include "ct_debug.h"
#include "geometry.h"
#include "memory.h"
#include <queue>
#include <Nutty.h>
#include <Copy.h>
#include <DeviceBuffer.h>

extern "C" void cudaCreateTriangleAABBs(CTreal3* tris, _AABB* aabbs, CTuint N);
extern "C" void cudaGetSceneBBox(nutty::DeviceBuffer<_AABB>& aabbs, CTuint N, _AABB& aabb);

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

void cpuTreeNode::Init(void)
{
    me = g_tree->m_linearNodeSplitAxis.Next();
    g_tree->m_linearNodeIsLeaf.Next();
    g_tree->m_linearNodeLeft.Next();
    g_tree->m_linearNodeRight.Next();
    g_tree->m_linearNodeSplits.Next();
    g_tree->m_linearNodePrimCount.Next();
    g_tree->m_linearNodePrimStartIndex.Next();
    g_tree->m_linearNodeToLeafIndex.Next();
}

CTuint cpuTreeNode::GetPrimStartIndex(void) const 
{
    return g_tree->m_linearNodePrimStartIndex[me];
}

void cpuTreeNode::SetLeaf(bool leaf)
{
    g_tree->m_linearNodeIsLeaf[me] = (byte)leaf;
}

void cpuTreeNode::SetSplit(CTreal s)
{
    g_tree->m_linearNodeSplits[me] = s;
}

void cpuTreeNode::SetPrimCount(CTuint c)
{
    g_tree->m_linearNodePrimCount[me] = c;
}

void cpuTreeNode::SetPrimStartIndex(CTuint i)
{
    g_tree->m_linearNodePrimStartIndex[me] = i;
}

void cpuTreeNode::SetSplitAxis(CT_SPLIT_AXIS axis)
{
    g_tree->m_linearNodeSplitAxis[me] = (byte)axis;
}

void cpuTreeNode::SetLeft(CTuint l)
{
    g_tree->m_linearNodeLeft[me] = l;
}

void cpuTreeNode::SetRight(CTuint r)
{
    g_tree->m_linearNodeRight[me] = r;
}

CTbool cpuTreeNode::IsLeaf(void) const
{
    return g_tree->m_linearNodeIsLeaf[me] != 0;
}

CTuint cpuTreeNode::LeftIndex(void) const
{
    return g_tree->m_linearNodeLeft[me];
}

CTuint cpuTreeNode::RightIndex(void) const
{
    return g_tree->m_linearNodeRight[me];
}

ICTTreeNode* cpuTreeNode::RightNode(void)
{
    return this + relRight;
}

ICTTreeNode* cpuTreeNode::LeftNode(void)
{
    return this + relLeft;
}

CTuint cpuTreeNode::GetPrimitiveCount(void) const
{
    return g_tree->m_linearNodePrimCount[me];
}

CT_SPLIT_AXIS cpuTreeNode::GetSplitAxis(void) const
{
    return (CT_SPLIT_AXIS)g_tree->m_linearNodeSplitAxis[me];
}

CTreal cpuTreeNode::GetSplit(void) const
{
    return g_tree->m_linearNodeSplits[me];
}

void cpuTreeNode::SetLeafIndex(CTuint i)
{
    g_tree->m_linearNodeToLeafIndex[me] = i;
}

CTuint cpuTreeNode::GetPrimitive(CTuint index) const
{
    return g_tree->GetPrimitive(GetPrimStartIndex() + index);
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

float g_split;
byte g_axis;

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

        if(parent->depth == m_depth || parent->GetPrimitiveCount() < 16)
        {
            parent->SetLeaf(true);
            parent->SetLeafIndex(m_leafNodesCount);
            for(CTuint i = 0; i < parent->GetPrimitiveCount(); ++i)
            {
                m_linearPerLeafNodePrimitives[m_linearPerLeafNodePrimitives.Next()] = parent->GetPrimitive(i);
            }
            m_linearPerLeafNodePrimCount.Append(parent->GetPrimitiveCount());
            m_linearPerLeafNodePrimStartIndex.Append(m_leafPrimOffset);
            m_leafPrimOffset += parent->GetPrimitiveCount();
            m_leafNodesCount++;
            continue;
        }

        m_interiorNodesCount++;
        AABB& parentAABB = m_linearNodeAABBs[parent->aabb];

        CT_SPLIT_AXIS axis = getLongestAxis(parentAABB.GetMax() - parentAABB.GetMin());
        m_events.clear();
        for(CTuint i = 0; i < parent->GetPrimitiveCount(); ++i)
        {
            _AABB& aabb = m_linearPrimAABBs[parent->GetPrimitive(i)];

            float start = getAxis(aabb._min, axis);
            float end = getAxis(aabb._max, axis);

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
            parent->SetLeaf(true);
            parent->SetLeafIndex(m_leafNodesCount);
            for(CTuint i = 0; i < parent->GetPrimitiveCount(); ++i)
            {
                m_linearPerLeafNodePrimitives[m_linearPerLeafNodePrimitives.Next()] = parent->GetPrimitive(i);
            }

            m_linearPerLeafNodePrimCount.Append(parent->GetPrimitiveCount());
            m_linearPerLeafNodePrimStartIndex.Append(m_leafPrimOffset);
            m_leafPrimOffset += parent->GetPrimitiveCount();
            m_leafNodesCount++;
            continue;
        }

        //ct_printf("Found best: sah=%f, axis=%d, split=%f\n\n\n ---------------------- \n\n\n", currentBestSAH, axis, currentSplit);
        parent->SetLeafIndex(m_leafNodesCount);
        parent->SetSplit(currentSplit);
        parent->SetSplitAxis(axis);

        CTuint l = m_linearNodeMemory.Next();
        CTuint r = m_linearNodeMemory.Next();   

        parent = &m_linearNodeMemory[nodeId];

        cpuTreeNode& leftNode = m_linearNodeMemory[l];
        cpuTreeNode& rightNode = m_linearNodeMemory[r];

        leftNode.Init();
        rightNode.Init();

        parent->SetLeft(l);
        parent->SetRight(r); 

        parent->relLeft = parent->LeftIndex() - parent->me;
        parent->relRight = parent->RightIndex() - parent->me;

        leftNode.aabb = m_linearNodeAABBs.Next();
        leftNode.SetSplit(0);
        leftNode.SetSplitAxis((CT_SPLIT_AXIS)-1);
        leftNode.SetLeaf(false);
        leftNode.me = parent->LeftIndex();

        m_linearNodeAABBs[parent->LeftIndex()] = m_linearNodeAABBs[parent->aabb];
        m_linearNodeAABBs[parent->LeftIndex()].ShrinkMax(axis, currentSplit);

        rightNode.aabb = m_linearNodeAABBs.Next();
        rightNode.SetSplit(0);
        rightNode.SetSplitAxis((CT_SPLIT_AXIS)-1);
        rightNode.SetLeaf(false);
        rightNode.me = parent->RightIndex();

        m_linearNodeAABBs[parent->RightIndex()] = m_linearNodeAABBs[parent->aabb];
        m_linearNodeAABBs[parent->RightIndex()].ShrinkMin(axis, currentSplit);

        parent->SetLeaf(false);

        CTuint start = m_linearPerNodePrimitives.Size();

        leftNode.SetPrimStartIndex(start);
        CTuint primCountLeft = 0;
        g_split = currentSplit;
        g_axis = axis;
        rightNode.SetPrimStartIndex(start);
        CTuint primCountRight = 0;

        m_linearSplitLeft.Reset();
        m_linearSplitRight.Reset();

        for(CTuint i = 0; i < parent->GetPrimitiveCount(); ++i)
        {
            _AABB& aabb = m_linearPrimAABBs[parent->GetPrimitive(i)];

            float mini = getAxis(aabb._min, axis);
            float maxi = getAxis(aabb._max, axis);

            if(mini < currentSplit)
            {
                primCountLeft++;
                CTuint id = m_linearSplitLeft.Next();
                m_linearSplitLeft[id] = parent->GetPrimitive(i);
            }

            if(maxi > currentSplit)
            {
                primCountRight++;
                CTuint id = m_linearSplitRight.Next();
                m_linearSplitRight[id] = parent->GetPrimitive(i);
            }
        }

        for(CTuint i = 0; i < primCountLeft; ++i)
        {
            CTuint id = m_linearPerNodePrimitives.Next();
            m_linearPerNodePrimitives[id] = m_linearSplitLeft[i];
        }
        for(CTuint i = 0; i < primCountRight; ++i)
        {
            CTuint id = m_linearPerNodePrimitives.Next();
            m_linearPerNodePrimitives[id] = m_linearSplitRight[i];
        }

        leftNode.SetPrimCount(primCountLeft);
        rightNode.SetPrimCount(primCountRight);
        rightNode.SetPrimStartIndex(start + primCountLeft);

        leftNode.depth = parent->depth + 1;
        rightNode.depth = parent->depth + 1;

        queue.push(parent->LeftIndex());
        queue.push(parent->RightIndex());
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

const void* cpuKDTree::GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const
{
    switch(type)
    {
    case eCT_LEAF_NODE_PRIM_IDS :
        {
            *byteCount = m_linearPerLeafNodePrimitives.Bytes();
            return m_linearPerLeafNodePrimitives.Begin();
        } break;
    case eCT_LEAF_NODE_PRIM_START_INDEX :
        {
            *byteCount = m_linearPerLeafNodePrimStartIndex.Bytes();
            return m_linearPerLeafNodePrimStartIndex.Begin();
        } break;
    case eCT_LEAF_NODE_PRIM_COUNT :
        {
            *byteCount = m_linearPerLeafNodePrimCount.Bytes();
            return m_linearPerLeafNodePrimCount.Begin();
        } break;
    case eCT_NODE_PRIM_IDS :
        {
            *byteCount = m_linearPerNodePrimitives.Bytes();
            return m_linearPerNodePrimitives.Begin();
        } break;
    case eCT_PRIMITVES :
        {
            return GetRawPrimitives(byteCount);
        }
    case eCT_NODE_SPLITS :
        {
            *byteCount = m_linearNodeSplits.Bytes();
            return m_linearNodeSplits.Begin();
        }
    case eCT_NODE_SPLIT_AXIS :
        {
            *byteCount = m_linearNodeSplitAxis.Bytes();
            return m_linearNodeSplitAxis.Begin();
        }
    case eCT_NODE_RIGHT_CHILD :
        {
            *byteCount = m_linearNodeRight.Bytes();
            return m_linearNodeRight.Begin();
        }
    case eCT_NODE_LEFT_CHILD :
        {
            *byteCount = m_linearNodeLeft.Bytes();
            return m_linearNodeLeft.Begin();
        }
    case eCT_NODE_IS_LEAF :
        {
            *byteCount = m_linearNodeIsLeaf.Bytes();
            return m_linearNodeIsLeaf.Begin();
        }
    case eCT_NODE_PRIM_COUNT :
        {
            *byteCount = m_linearNodePrimCount.Bytes();
            return m_linearNodePrimCount.Begin();
        }
    case eCT_NODE_PRIM_START_INDEX :
        {
            *byteCount = m_linearNodePrimStartIndex.Bytes();
            return m_linearNodePrimStartIndex.Begin();
        }
    case eCT_NODE_TO_LEAF_INDEX :
        {
            *byteCount = m_linearNodeToLeafIndex.Bytes();
            return m_linearNodeToLeafIndex.Begin();
        }
    default : *byteCount = 0; return NULL;
    }
}

void cpuKDTree::_DebugDrawNodes(CTuint parent, ICTTreeDebugLayer* dbLayer) const
{
    if(m_linearNodeIsLeaf.Get(parent))
    {
//         dbLayer->SetDrawColor(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
//         for(int i = 0; i < parent->geometry.size(); ++i)
//         {
//             dbLayer->DrawBox(geometry[parent->geometry[i]]->GetAABB());
//         }
        return;
    }

    CT_SPLIT_AXIS axis = (CT_SPLIT_AXIS)m_linearNodeSplitAxis.Get(parent);
    CTreal split = m_linearNodeSplits.Get(parent);

    CTreal3 start;
    CTreal3 end;

    setAxis(start, axis, split);
    setAxis(end, axis, split);

    CTint other0 = (axis + 1) % 3;
    CTint other1 = (axis + 2) % 3;

    //dbLayer->SetDrawColor(0,0.2f,0);

    AABB aaba = m_linearNodeAABBs.Get(parent);

    setAxis(start, other0, getAxis(aaba.GetMin(), other0));
    setAxis(start, other1, getAxis(aaba.GetMin(), other1));
    setAxis(end, other0, getAxis(aaba.GetMax(), other0));
    setAxis(end, other1, getAxis(aaba.GetMin(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(aaba.GetMin(), other0));
    setAxis(start, other1, getAxis(aaba.GetMin(), other1));
    setAxis(end, other0, getAxis(aaba.GetMin(), other0));
    setAxis(end, other1, getAxis(aaba.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(aaba.GetMin(), other0));
    setAxis(start, other1, getAxis(aaba.GetMax(), other1));
    setAxis(end, other0, getAxis(aaba.GetMax(), other0));
    setAxis(end, other1, getAxis(aaba.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    setAxis(start, other0, getAxis(aaba.GetMax(), other0));
    setAxis(start, other1, getAxis(aaba.GetMin(), other1));
    setAxis(end, other0, getAxis(aaba.GetMax(), other0));
    setAxis(end, other1, getAxis(aaba.GetMax(), other1));
    dbLayer->DrawLine(start, end);

    CTuint left = m_linearNodeLeft.Get(parent);
    CTuint right = m_linearNodeRight.Get(parent);
     _DebugDrawNodes(left, dbLayer);
     _DebugDrawNodes(right, dbLayer);
}

CT_RESULT cpuKDTree::DebugDraw(ICTTreeDebugLayer* dbLayer) const
{
    checkState(m_build);
    for(CTint i = 0; i < m_linearPrimAABBs.Size(); ++i)
    {
        _AABB& aabb = m_linearPrimAABBs.Get(i);

        float mini = getAxis(aabb._min, g_axis);
        float maxi = getAxis(aabb._max, g_axis);

        if(maxi > g_split)
        {
            AABB _aabb;
            _aabb.AddVertex(aabb._max);
            _aabb.AddVertex(aabb._min);
            dbLayer->DrawWiredBox(_aabb);
        }
    }
   
    dbLayer->DrawWiredBox(m_linearNodeAABBs.Get(m_root));
 
    _DebugDrawNodes(m_root, dbLayer);
    return CT_SUCCESS;
}

CTuint GenerateDepth(CTuint N)
{
    return N > 32 ? (CTuint)(8.5 + 1.3 * log(N)) : 1;
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
    m_linearNodeSplitAxis.Reset();
    m_linearNodeIsLeaf.Reset();
    m_linearNodeLeft.Reset();
    m_linearNodeRight.Reset();
    m_linearNodeSplits.Reset();
    m_linearNodePrimCount.Reset();
    m_linearNodePrimStartIndex.Reset();
    m_linearPerLeafNodePrimStartIndex.Reset();
    m_linearPerLeafNodePrimCount.Reset();
    m_linearNodeToLeafIndex.Reset();
    m_linearPerLeafNodePrimitives.Reset();

    m_leafPrimOffset = 0;
    m_leafNodesCount = 0;
    m_interiorNodesCount = 0;
    m_address = 0;
    m_linearNodeMemory.Reset();
    m_linearNodeAABBs.Reset();
    g_tree = this;

    m_root = m_linearNodeMemory.Next();
    
    cpuTreeNode& root = m_linearNodeMemory[m_root];
    root.Init();
    root.aabb = m_linearNodeAABBs.Next();
    m_linearNodeAABBs[root.aabb] = m_sceneAABB;
    root.SetPrimCount((CTuint)(m_linearPrimitiveMemory.size()/3));
    root.SetPrimStartIndex(0);
    root.me = 0;
    root.depth = 0;
    m_linearPerNodePrimitives.Reset();
    for(CTuint i = 0; i < root.GetPrimitiveCount(); ++i)
    {
        m_linearPerNodePrimitives[m_linearPerNodePrimitives.Next()] = i;
    }

    m_depth = min(64, max(1, (m_depth == -1 ? GenerateDepth((uint)root.GetPrimitiveCount()) : m_depth)));
    
    //_CreateTree(&root, 0);
    _CreateTree();
    m_build = true;
    return CT_SUCCESS;
}

CT_RESULT cpuKDTree::AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle)
{

    CTuint startindex = (CTuint)m_linearPrimitiveMemory.size();
    m_linearPrimitiveMemory.resize(m_linearPrimitiveMemory.size() + elements);
    m_originalLinearPrimitiveMemory.resize(m_originalLinearPrimitiveMemory.size() + elements);

    memcpy(&m_linearPrimitiveMemory[0] + startindex, (const CTreal3*)memory, sizeof(CTreal3) * elements);
    memcpy(&m_originalLinearPrimitiveMemory[0] + startindex, (const CTreal3*)memory, sizeof(CTreal3) * elements);
    CTuint aabbStart = m_linearPrimAABBs.Size();
    m_linearPrimAABBs.Resize(aabbStart + elements/3);

//     if(elements > 768)
//     {
//         nutty::DeviceBuffer<CTreal3> triangles(elements);
//         nutty::DeviceBuffer<_AABB> aabbs(elements/3);
//         nutty::HostBuffer<_AABB> h_aabbs(elements/3);
//         nutty::cuda::Copy(triangles.Begin()(), (CTreal3*)memory, elements, cudaMemcpyHostToDevice);
//         cudaCreateTriangleAABBs(triangles.Begin()(), aabbs.Begin()(), elements/3);
//         _AABB aabb;
//         cudaGetSceneBBox(aabbs, elements/3, aabb);
//         m_sceneAABB.AddVertex(aabb._min);
//         m_sceneAABB.AddVertex(aabb._max);
// 
//         nutty::Copy(h_aabbs.Begin(), aabbs.Begin(), aabbs.End());
// 
//         memcpy(m_linearPrimAABBs.Begin() + aabbStart, h_aabbs.Begin()(), (elements/3) * sizeof(_AABB));
// 
//         m_linearPrimAABBs.Advance(elements/3);
//     }
//     else
    {
        AABB aabb;
        for(CTuint i = 0; i < elements/3; ++i)
        {
            CTuint primAABB = m_linearPrimAABBs.Next();
            aabb.Reset();
            for(CTuint j = 0; j < 3; ++j)
            {
                const CTreal3& v = *((const CTreal3*)memory + 3*i + j);
                aabb.AddVertex(v);
                m_sceneAABB.AddVertex(v);
                m_linearPrimAABBs[primAABB].AddVertex(v);
            }
        }
    }

    CTulong h = (CTulong)m_linearGeoHandles.size();
    GeometryRange range;
    range.start = startindex;
    range.end = startindex + elements/3;
    m_handleRangeMap.insert(std::pair<CTulong, GeometryRange>(h, range));
    m_linearGeoHandles.push_back(h);
    *handle = (CTGeometryHandle)m_linearGeoHandles.back();
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
            m_linearPrimitiveMemory.reserve(m_linearPrimitiveMemory.size() + 3 * pc);
            for(CTuint i = 0; i < pc; ++i)
            {
                const CTTriangle* tri = static_cast<const CTTriangle*>(prims[i]);
                CTuint primAABB = m_linearPrimAABBs.Next();
                for(CTuint j = 0; j < 3; ++j)
                {
                    CTreal3 v;
                    tri->GetValue(j, v);
                    m_linearPrimitiveMemory.push_back(v);
                    m_originalLinearPrimitiveMemory.push_back(v);
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
    CTreal4 res;
    CTreal3 v = make_float3(vector.x, vector.y, vector.z);
    res.x = dot(make_float3(m3x3l[0].x, m3x3l[0].y, m3x3l[0].z), v);
    res.y = dot(make_float3(m3x3l[1].x, m3x3l[1].y, m3x3l[1].z), v);
    res.z = dot(make_float3(m3x3l[2].x, m3x3l[2].y, m3x3l[2].z), v);
    res.x += m3x3l[3].x;
    res.y += m3x3l[3].y;
    res.z += m3x3l[3].z;
    return make_float3(res.x, res.y, res.z);
}

void cpuKDTree::TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix)
{
    auto range = m_handleRangeMap.find(handle);
    if(range != m_handleRangeMap.end())
    {
         for(CTuint i = 0; i < range->second.end - range->second.start; ++i)
         {
             CTreal d = 0.01f;
             CTreal3 a = m_originalLinearPrimitiveMemory[range->second.start + 3 * i + 0];
             CTreal3 b = m_originalLinearPrimitiveMemory[range->second.start + 3 * i + 1];
             CTreal3 c = m_originalLinearPrimitiveMemory[range->second.start + 3 * i + 2];

             a = transform3f(matrix, a);
             b = transform3f(matrix, b);
             c = transform3f(matrix, c);

             m_linearPrimitiveMemory[range->second.start + 3 * i + 0] = a;
             m_linearPrimitiveMemory[range->second.start + 3 * i + 1] = b;
             m_linearPrimitiveMemory[range->second.start + 3 * i + 2] = c;
             
             _AABB& aabb = m_linearPrimAABBs[range->second.start/3 + i];
             aabb.Reset();
             aabb.AddVertex(a);
             aabb.AddVertex(b);
             aabb.AddVertex(c);

             m_sceneAABB.AddVertex(a);
             m_sceneAABB.AddVertex(b);
             m_sceneAABB.AddVertex(c);
         }
    }
}

CT_RESULT cpuKDTree::Init(CTuint flags /* = 0 */)
{
    m_build = false;
    return CT_SUCCESS;
}

cpuKDTree::~cpuKDTree(void)
{

}