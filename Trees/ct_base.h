#pragma once
#include "ct_def.h"
#include <host_defines.h>

typedef const char* CTuuid;

template <
    typename T
>
CTuuid __uuid(void)
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
        return QueryInterface(__uuidof(T), (void**)pp);
    }

    virtual CT_RESULT QueryInterface(CTuuid id, void** ppInterface) 
    { 
        *ppInterface = 0; 
        return CT_INTERFACE_NOT_FOUND; 
    }

    virtual ~ICTInterface(void) { }

    add_uuid_header(ICTInterface);
};

