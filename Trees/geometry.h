#pragma once
#include <vector>
#include "ct_runtime.h"
#include "ct_primitive.h"
#include "memory.h"
#include "check_state.h"
#include "output.h"

class ICTGeometry : public ICTInterface
{
public:
    virtual CT_GEOMETRY_TOPOLOGY GetTopology(void) const = 0;

    virtual const ICTPrimitive* const* GetPrimitives(CTuint* count) const = 0;

    virtual CT_RESULT AddPrimitive(const ICTPrimitive* prim) = 0;

    virtual CT_RESULT Transform(chimera::util::Mat4* matrix) = 0;

    virtual ~ICTGeometry(void) {}
};

//Implementations

class AABB : public ICTAABB
{
private:
    CTreal3 m_min;
    CTreal3 m_max;

    void _Shrink(CTbyte axis, CTbyte minMax, float v)
    {
        CTreal3& va = minMax ? m_max : m_min;
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
        Reset();
    }

    void Reset(void)
    {
        m_min.x = FLT_MAX;
        m_min.y = FLT_MAX;
        m_min.z = FLT_MAX;

        m_max.x = -FLT_MAX;
        m_max.y = -FLT_MAX;
        m_max.z = -FLT_MAX;
    }

    void AddVertex(const CTreal3& p)
    {
        m_min.x = fminf(p.x, m_min.x);
        m_min.y = fminf(p.y, m_min.y);
        m_min.z = fminf(p.z, m_min.z);

        m_max.x = fmaxf(p.x, m_max.x);
        m_max.y = fmaxf(p.y, m_max.y);
        m_max.z = fmaxf(p.z, m_max.z);
    }

    const CTreal3& GetMin(void) const
    {
        return m_min;
    }

    const CTreal3& GetMax(void) const
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

    AABB& operator=(const AABB& cpy)
    {
        m_max = cpy.m_max;
        m_min = cpy.m_min;
        return *this;
    }

    ~AABB(void)
    {

    }
};

class Geometry : public ICTGeometry
{
private:
    std::vector<const ICTPrimitive*> m_pPrimitives;
    ICTTree* m_pTree;

public:
    Geometry(void) : m_pTree(NULL), m_pPrimitives(0)
    {
    }

    void SetTree(ICTTree* tree)
    {
        m_pTree = tree;
    }

    CT_GEOMETRY_TOPOLOGY GetTopology(void) const
    {
        return CT_TRIANGLES;
    }

    const ICTPrimitive* const* GetPrimitives(CTuint* count) const
    {
        *count = (CTuint)m_pPrimitives.size();
        return &m_pPrimitives[0];
    }

    CT_RESULT AddPrimitive(const ICTPrimitive* prim)
    {
        switch(prim->GetType())
        {
        case CT_TRIANGLE:
            {
                const CTTriangle* tri = static_cast<const CTTriangle*>(prim);

                CTTriangle* triCpy = CTMemAllocObject<CTTriangle>();
                CTreal3 v;
                tri->GetValue(0, v);
                triCpy->SetValue(0, v);

                tri->GetValue(1, v);
                triCpy->SetValue(1, v);

                tri->GetValue(2, v);
                triCpy->SetValue(2, v);

                m_pPrimitives.push_back(triCpy);
                return CT_SUCCESS;
            } break;
        default : return CT_INVALID_ENUM;
        }
    }

     CT_RESULT Transform(chimera::util::Mat4* matrix);

    Geometry::~Geometry(void)
    {
        for(auto& it : m_pPrimitives)
        {
            CTMemFreeObject(it);
        }
    }
};

//---gpu stuff

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