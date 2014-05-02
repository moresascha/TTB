#pragma once
#include "ct_base.h"

class ICTMemoryView : public ICTInterface
{
public:
    virtual void* GetMemory(void) = 0;

    ~ICTMemoryView(void) { }

    add_uuid_header(ICTMemoryView);
};