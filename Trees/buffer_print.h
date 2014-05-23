#pragma once
#include "ct_types.h"
#include <DeviceBuffer.h>
#include <HostBuffer.h>

struct IndexedSAHSplit;
struct IndexedEvent;

std::ostream& operator<<(std::ostream &out, const IndexedSAHSplit& t);

std::ostream& operator<<(std::ostream &out, const IndexedEvent& t);

std::ostream& operator<<(std::ostream &out, const BBox& t);

std::ostream& operator<<(std::ostream &out, const AABB& t);

std::ostream& operator<<(std::ostream &out, const CTbyte& t);

template <
    typename Buffer
>
void PrintBuffer(const Buffer& buffer, size_t max = -1, const char* trim = " ")
{
    std::stringstream ss;
    for(CTuint i = 0; i < min(max, buffer.Size()); ++i)
    {
        ss << buffer[i] << /*"[" << i << "]" <<*/ " " << trim;
    }
    ss << "\n";
    OutputDebugStringA(ss.str().c_str());
}