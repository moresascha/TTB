#include <cutil_math.h>
#include "geometry.cuh"
#include "material.cuh"
#include "vector_functions.cuh"

__device__ Real3 _getTrianglelNormal(const Normal* norm, const Real2& bary)
{
    Real3 n = norm[1] * bary.x + norm[2] * bary.y + norm[0] * (1 - bary.x - bary.y);
    return normalize(n);
}

__device__ Real2 _getTrianglelTC(const TexCoord* tc, const Real2& bary)
{
    return tc[1] * bary.x + tc[2] * bary.y + tc[0] * (1 - bary.x - bary.y);
}

__device__ Real3 _getTrianglelHitPos(const Position* tc, const Real2& bary)
{
    return tc[1] * bary.x + tc[2] * bary.y + tc[0] * (1 - bary.x - bary.y);
}

__device__ Normal Triangles::getTrianglelNormal(uint index, const Real2& bary) const
{
    return _getTrianglelNormal(getGeometry().normals + 3 * index, bary);
}

__device__ TexCoord Triangles::getTrianglelTC(uint index, const Real2& bary) const
{
    return _getTrianglelTC(getGeometry().texCoords + 3 * index, bary);
}

__device__ Position Triangles::getTrianglelHitPos(uint index, const Real2& bary) const
{
    return _getTrianglelHitPos(getGeometry().positions + 3 * index, bary);
}

__device__ byte Triangles::getMaterialIndex(uint index) const
{
    return *(getGeometry().matId + 3 * index);
}

__device__ Material Triangles::getMaterial(uint index) const
{
    return getGeometry().materials[getMaterialIndex(index)];
}

__host__ void BBox::init(void)
{
    _min.x = FLT_MAX;
    _min.y = FLT_MAX;
    _min.z = FLT_MAX;

    _max.x = FLT_MIN;
    _max.y = FLT_MIN;
    _max.z = FLT_MIN;
}

__host__ void BBox::addPoint(const Real3& point)
{
    _min.x = min(_min.x, point.x - RAY_BBOX_DELTA);
    _min.y = min(_min.y, point.y - RAY_BBOX_DELTA);
    _min.z = min(_min.z, point.z - RAY_BBOX_DELTA);

    _max.x = max(_max.x, point.x + RAY_BBOX_DELTA);
    _max.y = max(_max.y, point.y + RAY_BBOX_DELTA);
    _max.z = max(_max.z, point.z + RAY_BBOX_DELTA);
}

__device__ __host__ float BBox::get(byte axis, byte mm) const
{
    switch(mm)
    {
        case 0 : return getAxis(_min, axis);
        case 1 : return getAxis(_max, axis);
    }
    return 0;
}

__device__ __host__ Real BBox::getX(byte mm) const
{
    return get(0, mm);
}
    
__device__ __host__ Real BBox::getY(byte mm) const
{
    return get(1, mm);
}

__device__ __host__ Real BBox::getZ(byte mm) const
{
    return get(2, mm);
}