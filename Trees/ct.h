#pragma once
#include <windows.h>
#include "ct_def.h"
#include "ct_base.h"
#include "ct_error.h"
#include "ct_types.h"

class ICTAABB : public ICTInterface
{
public:
    ICTAABB(void) {}

    virtual void AddVertex(const CTreal3& v) = 0;

    virtual const CTreal3& GetMin(void) const = 0;

    virtual const CTreal3& GetMax(void) const = 0;

    virtual void ShrinkMax(CTbyte axis, CTreal v) = 0;

    virtual void ShrinkMin(CTbyte axis, CTreal v) = 0;

    virtual void Reset(void) = 0;

    virtual ~ICTAABB(void) {}
};

struct ICTTreeNode : public ICTInterface
{
    virtual CTbool IsLeaf(void) const = 0;

    virtual CTuint LeftIndex(void) const = 0;

    virtual CTuint RightIndex(void) const = 0;

    virtual ICTTreeNode* LeftNode(void) = 0;

    virtual ICTTreeNode* RightNode(void) = 0;

    virtual CTuint GetPrimitiveCount(void) const = 0;
 
    virtual CTuint GetPrimitive(CTuint index) const = 0;

    virtual CT_SPLIT_AXIS GetSplitAxis(void) const = 0;

    virtual CTreal GetSplit(void) const = 0;

    static CTuuid uuid(void) { return "ICTTreeNode"; }

    ~ICTTreeNode(void) {}
};

CT_EXPORT CT_RESULT CT_API CTInit
    (
    CTuint flags
    );

CT_EXPORT CT_RESULT CT_API CTRelease
    (
    void
    );

CT_EXPORT CT_RESULT CT_API CTCreateSAHKDTree
    (
    ICTTree** tree,
    CTuint flags = 0
    );

CT_EXPORT CT_RESULT CT_API CTCreateTree
    (
    ICTTree** tree,
    CT_TREE_DESC* desc
    );

CT_EXPORT CT_RESULT CT_API CTCreateGeometry
    (
    ICTGeometry** geo
    );

CT_EXPORT CT_RESULT CT_API CTGetPrimitiveCount
    (
    const ICTTree* tree, 
    CTuint* count
    );

CT_EXPORT CT_RESULT CT_API CTAddGeometry
    (
    ICTTree* tree,
    ICTGeometry* geo,
    CTGeometryHandle* handle
    );

CT_EXPORT CT_RESULT CT_API CTAddGeometryFromLinearMemory
    (
    ICTTree* tree,
    const void* memory,
    CTuint elements,
    CTGeometryHandle* handle
    );

CT_EXPORT CT_RESULT CT_API CTTransformGeometryHandle
    (
    ICTTree* tree,
    CTGeometryHandle handle,
    const CTreal4* matrix4x4
    );

CT_EXPORT CT_RESULT CT_API CTTreeDrawDebug
    (
    const ICTTree* tree,
    ICTTreeDebugLayer* debugger
    );

CT_EXPORT CT_RESULT CT_API CTTransformGeometry
    (
    ICTGeometry* geo,
    CTreal matrix[4][4]
    );

CT_EXPORT CT_RESULT CT_API CTAddPrimitive
    (
    ICTGeometry* geo,
    ICTPrimitive* prim
    );

CT_EXPORT CT_RESULT CT_API CTPreAlloc
    (
    ICTTree* tree, 
    CTuint bytes
    );

CT_EXPORT CT_RESULT CT_API CTUpdate
    (
    ICTTree* tree
    );

CT_EXPORT CT_RESULT CT_API CTGetDepth
    (
    const ICTTree* tree,
    CTuint* depth
    );

CT_EXPORT CT_RESULT CT_API CTGetLinearMemory
    (
    const ICTTree* tree,
    CTuint* byteCnt,
    const void** memory,
    CT_LINEAR_MEMORY_TYPE type
    );

CT_EXPORT CT_RESULT CT_API CTGetRawLinearMemory
    (
    const ICTTree* tree,
    CTuint* byteCnt,
    const void** memory
    );

CT_EXPORT CT_RESULT CT_API CTGetTreeDeviceType
    (
    const ICTTree* tree, 
    CT_TREE_DEVICE* type
    );

CT_EXPORT CT_RESULT CT_API CTGetAxisAlignedBB
    (
    const ICTTree* tree, 
    const ICTAABB** aabb
    );

CT_EXPORT CT_RESULT CT_API CTRayCastGeometry
    (
    const ICTTree* tree,
    const CTreal3& eye,
    const CTreal3& dir,
    CTGeometryHandle* handlePtr
    );

CT_EXPORT CT_RESULT CT_API CTGetLeafNodeCount
    (
    const ICTTree* tree,
    CTuint* count
    );

CT_EXPORT CT_RESULT CT_API CTGetInteriorNodeCount
    (
    const ICTTree* tree,
    CTuint* count
    );

CT_EXPORT CT_RESULT CT_API CTTraverse
    (
    ICTTree* tree,
    CT_TREE_TRAVERSAL order,
    OnNodeTraverse callBack,
    void* userData = NULL
    );

CT_EXPORT CT_RESULT CT_API CTReleaseObject
    (
    ICTInterface* obj
    );

CT_EXPORT CT_RESULT CT_API CTReleaseGeometry
    (
    ICTGeometry* geo
    );

CT_EXPORT CT_RESULT CT_API CTReleasePrimitive
    (
    ICTPrimitive* prim
    );

CT_EXPORT CT_RESULT CT_API CTReleaseTree
    (
    ICTTree* tree
    );