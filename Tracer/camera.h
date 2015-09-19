#pragma once
#include <cutil_inline.h>

class Camera
{
private:
    float3 m_eyePos;
    float3 m_sideDir;
    float3 m_upDir;
    float3 m_viewDir;
    float3 m_view[3];
    float4 m_iview[4];
    float4 m_projection[4];

    void ComputeView(void);

public:

    float m_phi, m_theta;

    Camera(void)
    {
        memset(&m_eyePos, '0', sizeof(float3));
        memset(&m_sideDir, '0', sizeof(float3));
        memset(&m_upDir, '0', sizeof(float3));
        memset(&m_viewDir, '0', sizeof(float3));

        m_phi = m_theta = 0;

        m_eyePos.z = -1;
        m_sideDir.x = 1;
        m_upDir.y = 1;
        m_viewDir.z = 1;
        
        ComputeView();

        ComputeProj(1, 1);
    }

    void ComputeProj(float aspect);

    void ComputeProj(int w, int h)
    {
        ComputeProj(w / (float)h);
    }

    const float3* GetView(void) const
    {
        return m_view;
    }

    const float4* GetIView(void) const
    {
        return m_iview;
    }

    const float4* GetProjection(void) const
    {
        return m_projection;
    }

    const float3& GetEye(void) const
    {
        return m_eyePos;
    }

    void Rotate(float dPhi, float dTheta);

    void Move(const float3& delta);

    void Move(float dx, float dy, float dz);

    void Camera::SetEyePos(const float3& eye);

    void LookAt(const float3& lookAt, const float3& eye);
};