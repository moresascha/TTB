#pragma once
#include "ct_runtime.h"
#include <vector>
#include "geometry.h"
#include "tree.h"
#include <map>
#include <assert.h>

class cpuKDTree;

template <
    typename T, 
    CTuint growStep = 32000
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

    void _Grow(CTuint newSize)
    {
        T* newMem = new T[newSize];
        if(linearMem)
        {
            memcpy(newMem, linearMem, nextIndex * sizeof(T));
            _Delete();
        }
        linearMem = newMem;
        size = newSize;
    }

public:
    LinearMemory(void) : linearMem(NULL), nextIndex(0), size(0)
    {
    }

    void Resize(CTuint newSize)
    {
        if(newSize >= size)
        {
            _Grow(newSize);
        }
    }

    void Advance(CTuint adv)
    {
        SetPosition(nextIndex + adv);
    }

    void SetPosition(CTuint pos)
    {
        if(pos <= size)
        {
            nextIndex = pos;
        }
#ifdef _DEBUG
        else
        {
            __debugbreak();
        }
#endif
    }

    CTuint Next(void)
    {
        if(nextIndex >= size)
        {
            _Grow(size + growStep);
        }
        return nextIndex++;
    }

    CTuint Append(const T& val)
    {
        CTuint id = Next();
        operator[](id) = val;
        return id;
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

    T Get(CTuint index) const
    {
        return *(linearMem + index);
    }

    T operator[](CTuint index) const
    {
        return *(linearMem + index);
    }

    CTuint Size(void) const
    {
        return nextIndex;
    }

    CTuint Bytes(void) const
    {
        return Size() * sizeof(T);
    }

    CTuint Capacity(void) const
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

struct cpuTreeNode : public ICTTreeNode
{
    CTuint me;
    CTuint relLeft;
    CTuint relRight;
    CTuint aabb;
    CTuint depth;

    cpuTreeNode(void)
    {
    }

    cpuTreeNode(const cpuTreeNode&)
    {
    }

    void Print(void)
    {
        ct_printf("left=%d right=%d me=%d relLeft=%d relRight=%d split=%f axis=%d aabb=%d pc=%d pstart=%d\n", 
            LeftIndex(), RightIndex(), me, relLeft, relRight, GetSplit(), GetSplitAxis(), aabb, GetPrimitiveCount(), GetPrimStartIndex());
    }

    void Init(void);

    void cpuTreeNode::SetLeaf(bool leaf);

    void SetSplit(CTreal s);

    void SetPrimCount(CTuint c);
    
    void SetPrimStartIndex(CTuint i);
    
    void SetSplitAxis(CT_SPLIT_AXIS axis);
    
    void SetLeft(CTuint l);
    
    void SetRight(CTuint r);

    void SetLeafIndex(CTuint i);
    
    CTbool IsLeaf(void) const; 

    CTuint LeftIndex(void) const; 

    CTuint RightIndex(void) const; 
    
    ICTTreeNode* RightNode(void);
    
    ICTTreeNode* LeftNode(void);

    CTuint GetPrimitiveCount(void) const; 
    
    CTuint GetPrimitive(CTuint index) const; 

    CTuint GetPrimStartIndex(void) const; 

    CT_SPLIT_AXIS GetSplitAxis(void) const; 
    
    CTreal GetSplit(void) const; 
    
    ~cpuTreeNode(void);
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

struct SAHResult
{
    CTreal sah;
    CTreal split;
    CT_SPLIT_AXIS axis;
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
    CTuint m_leafPrimOffset;
    CTbool m_build;

    LinearMemory<CTbyte> m_linearNodeSplitAxis;
    LinearMemory<CTbyte> m_linearNodeIsLeaf;
    LinearMemory<CTuint> m_linearNodeToLeafIndex;
    LinearMemory<CTuint> m_linearNodeLeft;
    LinearMemory<CTuint> m_linearNodeRight;
    LinearMemory<CTreal> m_linearNodeSplits;
    LinearMemory<CTuint> m_linearNodePrimCount;
    LinearMemory<CTuint> m_linearNodePrimStartIndex;

    LinearMemory<cpuTreeNode> m_linearNodeMemory;
    LinearMemory<AABB> m_linearNodeAABBs;
    LinearMemory<_AABB> m_linearPrimAABBs;
    LinearMemory<CTuint> m_linearPerNodePrimitives;

    LinearMemory<CTuint> m_linearPerLeafNodePrimitives;
    LinearMemory<CTuint> m_linearPerLeafNodePrimStartIndex;
    LinearMemory<CTuint> m_linearPerLeafNodePrimCount;

    LinearMemory<CTuint> m_linearSplitLeft;
    LinearMemory<CTuint> m_linearSplitRight;

    std::vector<EdgeEvent> m_events; //todo

    std::vector<CTulong> m_linearGeoHandles;
    std::vector<CTreal3> m_linearPrimitiveMemory;
    std::vector<CTreal3> m_originalLinearPrimitiveMemory;
    std::map<CTulong, GeometryRange> m_handleRangeMap;

    void _CreateTree(cpuTreeNode* parent, CTuint depth);

    void _CreateTree(void);

    void _DebugDrawNodes(CTuint parent, ICTTreeDebugLayer* dbLayer) const;

    SAHResult ComputeSAH(const AABB& aabb, cpuTreeNode* node);

protected:
    CTuint GetPrimitive(CTuint index);

public:
    cpuKDTree(void) : m_depth(-1), m_address(0), m_build(false)
    {
    }

    void OnGeometryMoved(const CTGeometryHandle geo);

    CT_RESULT RayCast(const CTreal3& eye, const CTreal3& dir, CTGeometryHandle* handle) const;

    const CTGeometryHandle* GetGeometry(CTuint* gc)
    {
        *gc = (CTuint)m_linearGeoHandles.size();
        return (CTGeometryHandle*)&m_linearGeoHandles[0];
    }

    CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo)
    {
        return CT_SUCCESS;
    }

    CTuint GetPrimitiveCount(void) const
    {
        return m_linearPrimAABBs.Size();
    }

    CT_RESULT Init(CTuint flags = 0);

    CT_RESULT Update(void);

    ICTTreeNode* GetRoot(void) const
    {
        return (ICTTreeNode*)&m_root;
    }

    void TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix);

    CT_RESULT AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle);

    CT_RESULT AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle);

    CT_RESULT DebugDraw(ICTTreeDebugLayer* dbLayer) const;

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
        *byteCnt = (CTuint)(m_linearPrimitiveMemory.size() * sizeof(CTreal3));
        return (CTreal*)&m_linearPrimitiveMemory[0];
    }

    const void* GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const;

    CT_RESULT Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData = NULL);

    ~cpuKDTree(void);

    add_uuid_header(cpuKDTree);
};