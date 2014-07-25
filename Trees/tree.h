#pragma once
#include "ct.h"
#include <float.h>
#include <driver_types.h>

class ICTTree : public ICTInterface
{
public:
    virtual void OnGeometryMoved(const CTGeometryHandle geo) = 0;

    virtual CT_RESULT SetTopology(CT_GEOMETRY_TOPOLOGY topo) = 0;

    virtual CT_RESULT Init(CTuint flags = 0) = 0;

    virtual CT_RESULT Update(void) = 0;

    virtual CT_RESULT DebugDraw(ICTTreeDebugLayer* dbLayer) const = 0;

    virtual CT_RESULT AddGeometry(ICTGeometry* geo, CTGeometryHandle* handle) = 0;

    virtual void SetDepth(CTbyte depth) = 0;

    virtual cudaStream_t GetStream(void) { return NULL; }

    virtual CTuint GetDepth(void) const = 0;

    virtual CTuint GetInteriorNodesCount(void) const = 0;

    virtual CT_RESULT RayCast(const CTreal3& eye, const CTreal3& dir, CTGeometryHandle* handle) const = 0;

    virtual CTuint GetLeafNodesCount(void) const = 0;

    virtual CT_TREE_DEVICE GetDeviceType(void) const = 0;

    virtual CT_GEOMETRY_TOPOLOGY GetTopology(void) const = 0;

    virtual const CTGeometryHandle* GetGeometry(CTuint* gc) = 0;

    virtual const CTreal* GetRawPrimitives(CTuint* pc) const = 0;

    virtual CTuint GetPrimitiveCount(void) const = 0;

    virtual CT_RESULT Traverse(CT_TREE_TRAVERSAL order, OnNodeTraverse cb, void* userData = NULL) = 0;

    virtual void TransformGeometry(CTGeometryHandle handle, const CTreal4* matrix) = 0;

    virtual const void* GetLinearMemory(CT_LINEAR_MEMORY_TYPE type, CTuint* byteCount) const = 0;

    virtual CT_RESULT AddGeometryFromLinearMemory(const void* memory, CTuint elements, CTGeometryHandle* handle) = 0;

    virtual const ICTAABB* GetAxisAlignedBB(void) const = 0;

    virtual ~ICTTree(void) {}

    static CTuuid uuid(void) { return "ICTTree"; }
};
