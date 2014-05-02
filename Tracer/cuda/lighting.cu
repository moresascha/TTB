#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include "material.cuh"

extern "C"__device__ float3 phongLighting(const float3& eye, const float3& world, const float3& lightPos, const float3& normal, const Material* material) 
{
    float3 posToEye = normalize(eye - world);

    float3 lightToPos = normalize(world - lightPos);

    float3 reflectVec = reflect(lightToPos, normal);

    float diffuse = saturate(dot(-lightToPos, normal));

    float specular = pow(saturate(dot(reflectVec, posToEye)), material->specularExp());
    
    return material->ambientI + material->diffuseI * diffuse + material->specularI * specular;
}

extern "C" __device__ Real3 _refract(const Real3& i, const Real3& n, Real eta)
{
    Real3 mi = i;
    Real cosi = dot(-mi, n);
    Real cost2 = 1.0f - eta * eta * (1.0f - cosi*cosi);
    Real3 t = eta * i + ((eta*cosi - sqrt(abs(cost2))) * n);
    return t * make_real3(cost2 > 0);
}

extern "C" __device__ Real3 refract(const Real3& I, const Real3& N, Real n1, Real n2)
{
    Real eta = n1 / n2;
    Real3 R;
    Real k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));
    if (k < 0.0)
        R = make_real3(0, 0, 0);
    else
        R = eta * I - (eta * dot(N, I) + sqrt(k)) * N;
    return R;
}

extern "C" __device__ Real reflectance(const Real3& n, const Real3& i, Real n1, Real n2)
{
    Real nq = n1 / n2;
    Real cosi = -dot(n, i);
    Real sint2 = nq * nq * (1.0 - cosi * cosi);

    if(sint2 > (Real)1.0) 
    { 
        return (Real)1.0; 
    }
    
    Real cost = sqrt(1.0 - sint2);
    Real r = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
    Real p = (n2 * cosi - n1 * cost) / (n2 * cosi + n1 * cost);

    return (r * r + p * p) / 2.0;
}

//reflectance approxi
extern "C" __device__ Real schlick(const Real3& n, const Real3& i, Real n1, Real n2, Real fresnel)
{
    Real r0 = (n1 - n2) / (n1 + n2);

    r0 *= r0;

    Real cosX = -dot(n, i);

    if(n1 > n2)
    {
        Real n = n1 / n2;
        Real sint2 = n * n * (1.0 - cosX * cosX);

        if(sint2 > 1.0)
        {
            return 1.0;
        }
        cosX = sqrt(1.0 - sint2);
    }

    Real x = 1 - cosX;
    return r0 + (1.0 - r0) * pow(x, fresnel);
}

extern "C" __device__ Real Reflectance(const Real3& i, const Real3& n, Real n1, Real n2, Real fresnel)
{
    return fresnel > 0 ? schlick(n, i, n1, n2, fresnel) : 1;
}