#include "buffer_print.h"
#include "cuKDTree.h"

std::ostream& operator<<(std::ostream &out, const IndexedSAHSplit& t)
{
    out << "SAH="<< (t.sah == FLT_MAX ? -1.0f : t.sah);//"[" << (t.sah == FLT_MAX ? -1.0f : t.sah) << ", " << t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const IndexedEvent& t)
{
    out << t.v; //"[Split=" << t.v /*"[" << t.v*/ << ", " << (CTuint)t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const BBox& t)
{
    out << "[" << t.GetMin().x << "," << t.GetMin().y << "," << t.GetMin().z << "|" << t.GetMax().x << "," << t.GetMax().y << "," << t.GetMax().z << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const AABB& t)
{
    out << "[" << t.GetMin().x << "," << t.GetMin().y << "," << t.GetMin().z << "|" << t.GetMax().x << "," << t.GetMax().y << "," << t.GetMax().z << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const CTbyte& t)
{
    out << (CTint)t;
    return out;
}

std::ostream& operator<<(std::ostream &out, const CTuint& t)
{
    out << (CTint)t;
    return out;
}