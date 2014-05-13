#pragma once
#include "ct_runtime.h"
#include <vector>
#include "geometry.h"
#include "tree.h"
#include <map>
#include <assert.h>

class cpuKDTree;

struct cpuTreeNode : public ICTTreeNode
{
    CTbool isLeaf;
    CTuint left;
    CTuint right;
    CTuint me;

    CTuint relLeft;
    CTuint relRight;

    CTreal split;
    CT_SPLIT_AXIS splitAxis;
    CTuint aabb;
    CTbool visited;
    
    CTuint primCount;
    CTuint primStartIndex;

    CTuint depth;

    cpuTreeNode(void) : left(NULL), right(NULL), isLeaf(false), visited(false), aabb(NULL), relLeft(0), relRight(0)
    {
    }

    cpuTreeNode(const cpuTreeNode&)
    {
        //
    }

    void Print(void)
    {
        ct_printf("left=%d right=%d me=%d relLeft=%d relRight=%d split=%f axis=%d aabb=%d pc=%d pstart=%d\n", 
            left, right, me, relLeft, relRight, split, splitAxis, aabb,primCount, primStartIndex);
    }

    CTbool IsLeaf(void)
    {
        return isLeaf;
    }

    CTuint LeftIndex(void)
    {
        return left;
    }

    CTuint RightIndex(void)
    {
        return right;
    }

    ICTTreeNode* RightNode(void)
    {
        return this + relRight;
    }

    ICTTreeNode* LeftNode(void)
    {
        return this + relLeft;
    }

    CTuint GetPrimitiveCount(void)
    {
        return primCount;
    }

    CTuint GetPrimitive(CTuint index);

    CT_SPLIT_AXIS GetSplitAxis(void)
    {
        return splitAxis;
    }

    CTreal GetSplit(void)
    {
        return split;
    }

    ~cpuTreeNode(void);
};

struct GeometryRange
{
    CTuint start;
    CTuint end;
};

template <
    typename T, 
    CTuint growStep = 2048
>
class LinearMemory
{
private:
    T* linearMem;
    CTuint nextIndex;
    CTuint size;

    void _Delete(void)
    {
        if(linearMem)
        {
            delete[] linearMem;
            linearMem = NULL;
        }
    }

    void _Grow(void)
    {
        T* newMem = new T[growStep + size];
        if(linearMem)
        {
            memcpy(newMem, linearMem, nextIndex * sizeof(T));
            _Delete();
        }
        linearMem = newMem;
        size = growStep + size;
    }

public:
    LinearMemory(void) : linearMem(NULL), nextIndex(0), size(0)
    {
    }

    CTuint Next(void)
    {
        if(nextIndex >= size)
        {
            _Grow();
        }
        return nextIndex++;
    }

    void Reset(void)
    {
        nextIndex = 0;
    }

    T& operator[](CTuint index)
    {
#ifdef _DEBUG
        if(index > size)
        {
            __debugbreak();
        }
#endif
        return *(linearMem + index);
    }

    CTuint Size(void)
    {
        return nextIndex;
    }

    CTuint Capacity(void)
    {
        return size;
    }

    T* Begin(void) const
    {
        return linearMem;
    }

    ~LinearMemory(void)
    {
        _Delete();
    }
};

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
        return split < o.split;
    }
};

class cpuKDTree : public ICTTree
{
    friend struct cpuTreeNode;
private:
    CTuint m_depth;
    AABB m_sceneAABB;
    CTuint m_root;
    CTuint m_address;
    CTuint m_interiorNodesCount;
    CTuint m_leafNodesCount;

    LinearMemory<cpuTreeNode> m_linearNodeMemory;
    LinearMemory<AABB> m_linearNodeAABBs;
    LinearMemory<AABB> m_linearPrimAABBs;
    LinearMemory<CTuint> m_linearPerNodePrimitives;

    LinearMemory<CTuint> m_linearSplitLeft;
    LinearMemory<CTuint> m_linearSplitRight;

    std::vector<EdgeEvent> m_events; //todo

    std::vector<CTulong> m_linearGeoHandles;
    std::vector<CTreal3> m_linearPrimitiveMemory;
    std::map<CTulong, GeometryRange> m_handleRangeMap;

    void _CreateTree(cpuTreeNode* parent, CTuint depth);

    void _CreateTree(void);

protected:
    CTuint GetPrimitive(CTuint index);

public:
    cpuKDTree(void) : m_depth(-1), m_address(0)
    {
    }

    void OnGeometryMoved(const CTGeometryHandle geo);

    const CTGeometryHandle* GetGeometry(CTuint* gc)
    {
        *gc = (CTuint)m_linearGeoHandles.size();
        return (CTGeometryHandle*)&m_linearGeoHandles[0];
    }

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo)
    {
        return CT_SUCCESS;
    }

    CT_RESULT Init(CTuint flags = 0);

    CT_RESULT Update(void);

    ICTTreeNode* GetRoot(void) const
    {
        return (ICTTreeNode*)&m_root;
    }

    void TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix);

    CT_RESULT AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle);

    void DebugDraw(ICTTreeDebugLayer* dbLayer);

    uint GetDepth(void) const
    {
        return m_depth;
    }

    uint GetInteriorNodesCount(void) const;

    uint GetLeafNodesCount(void) const;

    void SetDepth(CTbyte depth)
    {
        m_depth = depth;
    }

    const ICTAABB* GetAxisAlignedBB(void) const
    {
        return m_linearNodeAABBs.Begin();
    }

    CT_TREE_DEVICE GetDeviceType(void) const
    {
        return eCT_CPU;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return CT_TRIANGLES;
    }

    const CTreal* GetRawPrimitives(CTuint* byteCnt) const
    {
        *byteCnt = m_linearPrimitiveMemory.size() * sizeof(CTreal3);
        return (CTreal*)&m_linearPrimitiveMemory[0];
    }

    CT_RESULT Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData = NULL);

    ~cpuKDTree(void);

    add_uuid_header(cpuKDTree);
};