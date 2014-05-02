#pragma once
#include "globals.cuh"

struct Triangles
{
    Material* materials;
    Position* positions;
    Normal* normals;
    TexCoord* texCoords;
    float3* colors;
    byte* matId;

    __device__ Normal getTrianglelNormal(uint index, const Real2& bary) const;
    __device__ TexCoord getTrianglelTC(uint index, const Real2& bary) const;
    __device__ Position getTrianglelHitPos(uint index, const Real2& bary) const;
    __device__ byte getMaterialIndex(uint index) const;
    __device__ Material getMaterial(uint index) const;
};

struct BBox
{
    Real3 _min;
    Real3 _max;

    __host__ void init(void);
    __host__ void addPoint(Real3& point);
    __device__ __host__ float get(byte axis, byte mm) const;
    __device__ __host__ Real getX(byte mm) const;
    __device__ __host__ Real getY(byte mm) const;
    __device__ __host__ Real getZ(byte mm) const;
};
