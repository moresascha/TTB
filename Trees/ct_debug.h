#pragma once
#include "ct.h"

class ICTTreeDebugLayer : public ICTInterface
{
public:
    virtual void DrawLine(const CTreal3& start, const CTreal3& end) = 0;

    virtual void DrawBox(const ICTAABB& aabb) = 0;

    virtual void DrawWiredBox(const ICTAABB& aabb) = 0;

    virtual void SetDrawColor(CTreal r, CTreal g, CTreal b) = 0;

    virtual ~ICTTreeDebugLayer(void) {}
};