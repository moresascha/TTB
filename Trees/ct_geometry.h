#pragma once
#include "ct_base.h"
#include "ct.h"

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

class ICTGeometry : public ICTInterface
{
public:
    virtual const ICTAABB& GetAABB(void) const = 0;

    virtual CT_GEOMETRY_TOPOLOGY GetTopology(void) const = 0;
    
    virtual const ICTVertex** GetVertices(uint* count) = 0;

    virtual void AddVertex(const ICTVertex* v) = 0;

    virtual ~ICTGeometry(void) {}
};