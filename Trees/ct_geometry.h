#pragma once
#include "ct_base.h"
#include "ct.h"
namespace chimera
{
    namespace util
    {
        class Mat4;
    }
}

class ICTVertex : public ICTInterface
{
public:
    virtual const ctfloat3& GetPosition(void) const = 0;

    virtual void SetPosition(const ctfloat3& pos) = 0;

    virtual ~ICTVertex(void) {}
};

class ICTAABB : public ICTInterface
{
public:
    virtual void AddVertex(const ICTVertex* v) = 0;

    virtual const ctfloat3& GetMin(void) const = 0;

    virtual const ctfloat3& GetMax(void) const = 0;

    virtual ~ICTAABB(void) {}
};

class ICTPrimitive : public ICTInterface
{
public:
    virtual const ICTVertex* const* GetVertices(uint* count) const = 0;

    virtual const ICTAABB& GetAABB(void) const = 0;

    virtual CT_GEOMETRY_TOPOLOGY GetTopology(void) const = 0;

    virtual CT_RESULT Transform(chimera::util::Mat4* matrix) = 0;

    virtual CT_RESULT AddVertex(ICTVertex* v) = 0;

    virtual ~ICTPrimitive(void) {}
};

class ICTGeometry : public ICTInterface
{
public:
    virtual CT_GEOMETRY_TOPOLOGY GetTopology(void) const = 0;
    
    virtual const ICTPrimitive* const* GetPrimitives(uint* count) const = 0;

    virtual CT_RESULT AddPrimitive(ICTPrimitive* prim) = 0;

    virtual CT_RESULT Transform(chimera::util::Mat4* matrix) = 0;

    virtual ~ICTGeometry(void) {}
};