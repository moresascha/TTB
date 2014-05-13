#include "cpuKDTree.h"
#include "output.h"
#include "ct_primitive.h"
#include "geometry.h"

CT_RESULT Geometry::Transform(chimera::util::Mat4* matrix)
{
    if(m_pTree)
    {
//         for(CTuint i = 0; i < m_pArrayVector.size(); ++i)
//         {
//             ICTPrimitive* prim = m_pArrayVector[i];
//             prim->Transform(matrix);
//         }

        //m_pTree->OnGeometryMoved(this);
        return CT_SUCCESS;
    }
    return CT_INVALID_OPERATION;
}