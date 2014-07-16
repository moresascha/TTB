#pragma once

//forward declaration
class ICTTree;
class ICTGeometry;
class ICTVertex;
class ICTAABB;
class ICTTreeDebugLayer;
class ICTPrimitive;
class CTTriangle;
struct float4;
struct ICTTreeNode;

struct float3;
struct float2;
struct uint3;
struct char3;

typedef unsigned char CTbyte;
typedef char3 CTbyte3;
typedef uint3 CTuint3;
typedef float3 CTreal3;
typedef float4 CTreal4;
typedef float2 CTreal2;
typedef float CTreal;
typedef bool CTbool;
typedef int CTint;
typedef unsigned int CTuint;
typedef long long CTlong;
typedef unsigned long long CTulong;

typedef CTlong CTGeometryHandle;

typedef void(*OnNodeTraverse)(ICTTreeNode* node, void* userData);

enum CT_GEOMETRY_TOPOLOGY
{
    CT_TRIANGLES = 0
};

enum CT_PRIMITIVE_TYPE
{
    CT_TRIANGLE = 0
};

enum CT_TREE_DEVICE
{
    eCT_CPU,
    eCT_GPU
};

enum CT_TREE_TYPE
{
    eCT_TreeTypeKD = 0
};

enum CT_SPLIT_STRATEGY
{
    eCT_SAH = 0
};

struct CT_TREE_DESC
{
    CT_TREE_TYPE type;
    CT_SPLIT_STRATEGY strategy;
    CTuint flags;
};

enum CT_SPLIT_AXIS
{
    eCT_X = 0,
    eCT_Y = 1,
    eCT_Z = 2
};

enum CT_TREE_TRAVERSAL
{
    eCT_DEPTH_FIRST,
    eCT_BREADTH_FIRST,
    eCT_VEB
};

enum CT_LINEAR_MEMORY_TYPE
{
    eCT_LEAF_NODE_PRIM_IDS = 1,
    eCT_LEAF_NODE_PRIM_START_INDEX = 2,
    eCT_LEAF_NODE_PRIM_COUNT = 3,

    eCT_NODE_PRIM_IDS = 4,
    eCT_NODE_IS_LEAF = 5,
    eCT_NODE_PRIM_START_INDEX = 6,
    eCT_NODE_PRIM_COUNT = 7,
    eCT_NODE_LEFT_CHILD = 8,
    eCT_NODE_RIGHT_CHILD = 10,
    eCT_NODE_SPLITS = 11,
    eCT_NODE_SPLIT_AXIS = 12,
    eCT_NODE_TO_LEAF_INDEX = 13,
    eCT_PRIMITVES = 14
};