#pragma once
#include <float.h>
#include <vector>
#include "ct_runtime.h"
#include "memory.h"
#include "check_state.h"

class cpuKDTree;

class AABB : public ICTAABB
{
private:
    ctfloat3 m_min;
    ctfloat3 m_max;

    void _Shrink(byte axis, byte minMax, float v)
    {
        ctfloat3& va = minMax ? m_max : m_min;
        switch(axis)
        {
        case 0 : va.x = v; return;
        case 1 : va.y = v; return;
        case 2 : va.z = v; return;
        }
    }

public:
    AABB(const AABB& aabb)
    {
        m_max = aabb.m_max;
        m_min = aabb.m_min;
    }

    AABB(void)
    {
        m_min.x = FLT_MAX;
        m_min.y = FLT_MAX;
        m_min.z = FLT_MAX;

        m_max.x = -FLT_MAX;
        m_max.y = -FLT_MAX;
        m_max.z = -FLT_MAX;
    }

    void AddVertex(const ICTVertex* v)
    {
        const ctfloat3& p = v->GetPosition();
        m_min.x = fminf(p.x, m_min.x);
        m_min.y = fminf(p.y, m_min.y);
        m_min.z = fminf(p.z, m_min.z);

        m_max.x = fmaxf(p.x, m_max.x);
        m_max.y = fmaxf(p.y, m_max.y);
        m_max.z = fmaxf(p.z, m_max.z);
    }

    const ctfloat3& GetMin(void) const
    {
        return m_min;
    }

    const ctfloat3& GetMax(void) const
    {
        return m_max;
    }

    void ShrinkMax(byte axis, float v)
    {
        _Shrink(axis, 1, v);
    }

    void ShrinkMin(byte axis, float v)
    {
        _Shrink(axis, 0, v);
    }
};

class Vertex : public ICTVertex
{
private:
    ctfloat3 m_position;

public:
    Vertex(void)
    {
        m_position.x = m_position.y = m_position.z = 0;
    }

    Vertex(ctfloat3& pos)
    {
        m_position.x = pos.x;
        m_position.y = pos.y;
        m_position.z = pos.z;
    }

    Vertex(const Vertex& v)
    {
        m_position.x = v.m_position.x;
        m_position.y = v.m_position.y;
        m_position.z = v.m_position.z;
    }

    const ctfloat3& GetPosition(void) const
    {
        return m_position;
    }

    void SetPosition(const ctfloat3& pos)
    {
        m_position.x = pos.x;
        m_position.y = pos.y;
        m_position.z = pos.z;
    }
};

class Triangle : public ICTPrimitive
{
private:
    ICTVertex* m_pArrayVector[3];
    AABB m_aabb;
    byte m_currentPos;
public:

    Triangle(void) : m_currentPos(0)
    {
        ZeroMemory(m_pArrayVector, 3 * sizeof(ICTVertex*));
    }

    const ICTAABB& GetAABB(void) const
    {
        return m_aabb;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return CT_TRIANGLES;
    }

    const ICTVertex* const* GetVertices(uint* count) const
    {
        *count = 3;
        return m_pArrayVector;
    }

    CT_RESULT AddVertex(ICTVertex* v)
    {
        checkState(m_currentPos < 3);
        m_pArrayVector[m_currentPos++] = v;
        m_aabb.AddVertex(v);
        return CT_SUCCESS;
    }

    CT_RESULT Transform(chimera::util::Mat4* matrix);

    Triangle::~Triangle(void)
    {
        for(auto& it : m_pArrayVector)
        {
            CTMemFreeObject(it);
        }
    }
};

class Geometry : public ICTGeometry
{
private:
    std::vector<ICTPrimitive*> m_pArrayVector;
    cpuKDTree* m_pTree;

public:
    Geometry(void) : m_pTree(NULL)
    {
    }

    void SetTree(cpuKDTree* tree)
    {
        m_pTree = tree;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return CT_TRIANGLES;
    }

    const ICTPrimitive* const* GetPrimitives(uint* count) const
    {
        *count = (uint)m_pArrayVector.size();
        return &(m_pArrayVector[0]);
    }

     CT_RESULT AddPrimitive(ICTPrimitive* prim)
    {
        checkState(prim->GetTopology() == GetTopology());

        m_pArrayVector.push_back(prim);
        return CT_SUCCESS;
    }

     CT_RESULT Transform(chimera::util::Mat4* matrix);

    Geometry::~Geometry(void)
    {
        for(auto& it : m_pArrayVector)
        {
            CTMemFreeObject(it);
        }
    }
};

//---gpu stuff

typedef Vertex GPUVertex;
typedef Triangle GPUTriangle;
typedef Geometry GPUGeometry;

// class GPUVertex : public ICTVertex
// {
// 
// };
// 
// class GPUTriangle : public ICTPrimitive
// {
// 
// };
// 
// class GPUGeometry : public ICTGeometry
// {
// };