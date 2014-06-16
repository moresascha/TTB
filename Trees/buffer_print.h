#pragma once
#include "ct_types.h"
#include <DeviceBuffer.h>
#include <HostBuffer.h>
#include "cuKDTree.h"

struct IndexedSAHSplit;
struct IndexedEvent;
struct dpNodeContent;

std::ostream& operator<<(std::ostream &out, const IndexedSAHSplit& t);

std::ostream& operator<<(std::ostream &out, const IndexedEvent& t);

std::ostream& operator<<(std::ostream &out, const dpNodeContent& t);

std::ostream& operator<<(std::ostream &out, const BBox& t);

std::ostream& operator<<(std::ostream &out, const AABB& t);

std::ostream& operator<<(std::ostream &out, const CTbyte& t);

std::ostream& operator<<(std::ostream &out, const CTbyte& t);

#define PRINT_BUFFER(_name) \
    OutputDebugStringA(#_name);\
    OutputDebugStringA(":  ");\
    PrintBuffer(_name);

#define PRINT_BUFFER_N(_name, _length) \
    OutputDebugStringA(#_name);\
    OutputDebugStringA(":  ");\
    PrintBuffer(_name, _length);


template <
    typename Buffer
>
void PrintBuffer(const Buffer& buffer, size_t max, const char* trim)
{
    std::stringstream ss;
    for(CTuint i = 0; i < min(max, buffer.Size()); ++i)
    {
        ss << buffer[i] /*<< "[" << i << "]" */ << " " << trim;
    }
    ss << "\n";
    OutputDebugStringA(ss.str().c_str());
}

template <
    typename Buffer
>
void PrintBuffer(const Buffer& buffer, size_t max)
{
    PrintBuffer(buffer, max, " ");
}

template <
    typename Buffer
>
void PrintBuffer(const Buffer& buffer)
{
    PrintBuffer(buffer, (size_t)-1, " ");
}