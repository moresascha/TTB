#pragma once
#include "globals.cuh"

extern "C" __device__ float4 readTexture(uint slot, const Real2& tc);