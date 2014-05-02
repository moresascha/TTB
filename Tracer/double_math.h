#pragma once
#include <cutil_inline.h>
#include <cutil_math.h>
#include <math_functions.h>

inline __host__ __device__ double dot(double3 a, double3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double dot(float3 a, double3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double3 normalize(double3 v)
{
    double invLen = rsqrt(dot(v, v));
    v.x *= invLen;
    v.y *= invLen;
    v.z *= invLen;
    return v;
}

inline __host__ __device__ double3 cross(double3 a, double3 b)
{ 
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    double3 v;
    v.x = a.x - b.x; v.y = a.y - b.y; v.z = a.z - b.z;
    return v;
}

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    double3 v;
    v.x = a.x + b.x; v.y = a.y + b.y; v.z = a.z + b.z;
    return v;
}

inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}