#pragma once
#include "ct.h"

namespace chimera
{
    namespace util
    {
        class Mat4;
    }
}

class ICTPrimitive : public ICTInterface
{
private:
    CT_PRIMITIVE_TYPE m_topo;

public:
    ICTPrimitive(CT_PRIMITIVE_TYPE topo) : m_topo(topo)
    {

    }

    CT_INLINE CT_PRIMITIVE_TYPE GetType(void) const
    {
        return m_topo;
    }

    virtual void GetAxisAlignedBB(ICTAABB& aabb) const = 0;
};

class CTTriangle : public ICTPrimitive
{
private:
    CTreal3 m_ABC[3];

public:
    CTTriangle(void) : ICTPrimitive(CT_TRIANGLE)
    {
    }

    CTTriangle(const CTTriangle& cpy) : ICTPrimitive(CT_TRIANGLE)
    {
        m_ABC[0] = cpy.m_ABC[0];
        m_ABC[1] = cpy.m_ABC[1];
        m_ABC[2] = cpy.m_ABC[2];
    }

    CT_RESULT GetValue(CTbyte index, CTreal3& v) const
    {
        if(index > 2)
        {
            return CT_INVALID_VALUE;
        }
        v = m_ABC[index];
        return CT_SUCCESS;
    }

    void SetValue(CTbyte index, const CTreal3& v)
    {
        if(index > 2)
        {
            return;
        }
        m_ABC[index] = v;
    }

    void GetAxisAlignedBB(ICTAABB& aabb) const
    {
        aabb.Reset();
        aabb.AddVertex(m_ABC[0]);
        aabb.AddVertex(m_ABC[1]);
        aabb.AddVertex(m_ABC[2]);
    }

    ~CTTriangle(void)
    {

    }
};
