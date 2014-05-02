#pragma once
#include "types.cuh"

#define FLT_MAX 3.402823466e+38F
#define FLT_MAX_DIV2 (FLT_MAX/2.0f)
#define PI 3.14159265359
#define PI_MUL2 (2 * PI)
#define RAY_BBOX_DELTA (Real)(1)
#define RAY_HIT_NORMAL_DELTA (Real)(0.005)
#define NO_TEXTURE (0x000000FF)
#define SUN_POS getSunPos()
#define BBOX_MAX getBBox()._max
#define BBOX_MIN getBBox()._min

extern "C" __device__ Real3 getSunPos(void);

extern "C" __device__ const BBox& getBBox(void);

extern "C" __device__ const Triangles& getGeometry(void);