#include "buffer_print.h"
#include "cuKDTree.h"

std::ostream& operator<<(std::ostream &out, const IndexedSAHSplit& t)
{
    out << "SAH="<< (t.sah == FLT_MAX ? -1.0f : t.sah);//"[" << (t.sah == FLT_MAX ? -1.0f : t.sah) << ", " << t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const IndexedEvent& t)
{
    out << "Split=" << t.v;//"[" << t.v << ", " << t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const BBox& t)
{
    out << "[" << t._min.x << "," << t._min.y << "," << t._min.z << "|" << t._max.x << "," << t._max.y << "," << t._max.z << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const AABB& t)
{
    out << "[" << t.GetMin().x << "," << t.GetMin().y << "," << t.GetMin().z << "|" << t.GetMax().x << "," << t.GetMax().y << "," << t.GetMax().z << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const CTbyte& t)
{
    out << (CTuint)t;
    return out;
}