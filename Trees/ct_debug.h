#pragma once
#include "ct.h"

class ICTTreeDebugLayer : public ICTInterface
{
public:
    virtual void DrawLine(const ctfloat3& start, const ctfloat3& end) = 0;

    virtual void DrawBox(const ICTAABB& aabb) = 0;

    virtual void DrawWiredBox(const ICTAABB& aabb) = 0;

    virtual void SetDrawColor(float r, float g, float b) = 0;

    virtual ~ICTTreeDebugLayer(void) {}
};