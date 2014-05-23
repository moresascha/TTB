#pragma once
#include <vector>
#include "ct_runtime.h"
#include "ct_primitive.h"
#include "memory.h"
#include "check_state.h"
#include "output.h"
#include "vec_functions.h"

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

#define BB_EPSILON 1e-3f

class AABB : public ICTAABB
{
private:
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
    CTreal3 m_min;
    CTreal3 m_max;

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
        m_min.x = fminf(p.x - BB_EPSILON, m_min.x);
        m_min.y = fminf(p.y - BB_EPSILON, m_min.y);
        m_min.z = fminf(p.z - BB_EPSILON, m_min.z);

        m_max.x = fmaxf(p.x + BB_EPSILON, m_max.x);
        m_max.y = fmaxf(p.y + BB_EPSILON, m_max.y);
        m_max.z = fmaxf(p.z + BB_EPSILON, m_max.z);
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
            }
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

struct _AABB
{
    CTreal3 _min;
    CTreal3 _max;

    __device__ __host__ _AABB(void)
    {
        Reset();
    }

    __device__ __host__ _AABB(const _AABB& aabb)
    {
        _min = aabb._min;
        _max = aabb._max;
    }

    __device__ __host__ ~_AABB(void)
    {

    }

    __device__ __host__  void Reset(void)
    {
        _min.x = FLT_MAX;
        _min.y = FLT_MAX;
        _min.z = FLT_MAX;

        _max.x = -FLT_MAX;
        _max.y = -FLT_MAX;
        _max.z = -FLT_MAX;
    }

    __device__ __host__ void AddVertex(const CTreal3& p)
    {
        _min.x = fminf(p.x - BB_EPSILON, _min.x);
        _min.y = fminf(p.y - BB_EPSILON, _min.y);
        _min.z = fminf(p.z - BB_EPSILON, _min.z);

        _max.x = fmaxf(p.x + BB_EPSILON, _max.x);
        _max.y = fmaxf(p.y + BB_EPSILON, _max.y);
        _max.z = fmaxf(p.z + BB_EPSILON, _max.z);
    }

    __device__ __host__ CTreal get(byte axis, byte mm) const
    {
        switch(mm)
        {
        case 0 : return getAxis(_min, axis);
        case 1 : return getAxis(_max, axis);
        }
        return 0;
    }

    __device__ __host__  CTreal getX(byte mm) const
    {
        return get(0, mm);
    }

    __device__ __host__  CTreal getY(byte mm) const
    {
        return get(1, mm);
    }

    __device__ __host__  CTreal getZ(byte mm) const
    {
        return get(2, mm);
    }

    __device__ __host__  const CTreal3& GetMin(void) const
    {
        return _min;
    }

    __device__ __host__ const CTreal3& GetMax(void) const
    {
        return _max;
    }
};

typedef _AABB BBox;