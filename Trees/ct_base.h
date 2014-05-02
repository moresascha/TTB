#pragma once
#include "ct_def.h"

typedef const char* ct_uuid;

template <
    typename T
>
ct_uuid __uuid(void)
{
    return T::uuid();
}

struct ICTInterface
{
    template <
        typename T
    >
    CT_RESULT QueryInterface(T** pp)
    {
        QueryInterface(__uuidof(T), (void**)pp);
    }

    virtual CT_RESULT QueryInterface(ct_uuid id, void** ppInterface) 
    { 
        *ppInterface = 0; 
        return CT_INTERFACE_NOT_FOUND; 
    }

    virtual ~ICTInterface(void) { }

    add_uuid_header(ICTInterface);
};