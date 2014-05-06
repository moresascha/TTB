#include "cpuKDTree.h"
#include "ct_output.h"

CT_RESULT Triangle::Transform(chimera::util::Mat4* matrix)
{
    for(uint i = 0; i < 3; ++i)
    {
        ICTVertex* v = m_pArrayVector[i];
        ctfloat3 p = v->GetPosition();
        p.y += 0.1f;
        v->SetPosition(p);
        m_aabb.AddVertex(v);
    }
    return CT_SUCCESS;
}

CT_RESULT Geometry::Transform(chimera::util::Mat4* matrix)
{
    if(m_pTree)
    {
        for(uint i = 0; i < m_pArrayVector.size(); ++i)
        {
            ICTPrimitive* prim = m_pArrayVector[i];
            prim->Transform(matrix);
        }

        m_pTree->OnGeometryMoved(this);
        return CT_SUCCESS;
    }
    return CT_INVALID_OPERATION;
}