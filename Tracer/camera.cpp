#include "camera.h"
#include <cutil_math.h>
#include <DirectXMath.h>

void Camera::ComputeProj(float aspect)
{
    float invTan = 1.0f / tan(DirectX::XM_PIDIV2 * 0.5f);
    float fnear = 1e-3f;
    float ffar = 1e3f;

    m_projection[0].x = invTan;
    m_projection[0].y = 0;
    m_projection[0].z = 0;
    m_projection[0].w = 0;

    m_projection[1].x = 0;
    m_projection[1].y = invTan * aspect;
    m_projection[1].z = 0;
    m_projection[1].w = 0;

    m_projection[2].x = 0;
    m_projection[2].y = 0;
    m_projection[2].z = (ffar + fnear) / (ffar - fnear);
    m_projection[2].w = 1;

    m_projection[3].x = 0;
    m_projection[3].y = 0;
    m_projection[3].z = ffar  * (1 - m_projection[2].z);
    m_projection[3].w = 0;
}

void Camera::ComputeView(void)
{
    float3 zAxis = normalize(m_viewDir);
    float3 yAxis = m_upDir;
    float3 xAxis = normalize(cross(yAxis, zAxis));
    yAxis = cross(zAxis, xAxis);

    m_view[0].x = xAxis.x;
    m_view[1].x = xAxis.y;
    m_view[2].x = xAxis.z;

    m_view[0].y = yAxis.x;
    m_view[1].y = yAxis.y;
    m_view[2].y = yAxis.z;

    m_view[0].z = zAxis.x;
    m_view[1].z = zAxis.y;
    m_view[2].z = zAxis.z;
    
    // for gl/d3d
    m_iview[0].x = xAxis.x;
    m_iview[1].x = xAxis.y;
    m_iview[2].x = xAxis.z;
    m_iview[3].x = -(m_eyePos.x * xAxis.x + m_eyePos.y * xAxis.y + m_eyePos.z * xAxis.z);

    m_iview[0].y = yAxis.x;
    m_iview[1].y = yAxis.y;
    m_iview[2].y = yAxis.z;
    m_iview[3].y = -(m_eyePos.x * yAxis.x + m_eyePos.y * yAxis.y + m_eyePos.z * yAxis.z);

    m_iview[0].z = zAxis.x;
    m_iview[1].z = zAxis.y;
    m_iview[2].z = zAxis.z;
    m_iview[3].z = -(m_eyePos.x * zAxis.x + m_eyePos.y * zAxis.y + m_eyePos.z * zAxis.z);

    m_iview[0].w = 0;
    m_iview[1].w = 0;
    m_iview[2].w = 0;
    m_iview[3].w = 1;
}

void Camera::Rotate(float dPhi, float dTheta)
{
    m_phi += dPhi;
    m_theta += dTheta;

    float sinPhi = sin(m_phi);
    float cosPhi = cos(m_phi);
    float sinTheta = sin(m_theta);
    float cosTheta = cos(m_theta);

    m_sideDir.x = cosPhi;
    m_sideDir.y = 0;
    m_sideDir.z = -sinPhi;

    m_upDir.x = sinPhi*sinTheta;
    m_upDir.y = cosTheta;
    m_upDir.z = cosPhi*sinTheta;

    m_viewDir.x = sinPhi*cosTheta;
    m_viewDir.y = -sinTheta;
    m_viewDir.z = cosPhi*cosTheta;

    ComputeView();
}

void Camera::Move(const float3& delta)
{
    Move(delta.x, delta.y, delta.z);
}

void Camera::Move(float dx, float dy, float dz)
{
    float3 deltaX(m_sideDir);
    deltaX *= dx;

    float3 deltaZ(m_viewDir);
    deltaZ *= dz;

    float3 deltaY(m_upDir);
    deltaY *= dy;

    deltaX += deltaY;
    deltaX += deltaZ;

    m_eyePos += deltaX; 

    ComputeView();
}

void Camera::SetEyePos(const float3& eye)
{
    m_eyePos = eye;
    ComputeView();
}

void Camera::LookAt(const float3& lookAt, const float3& eye)
{
    float3 dir = lookAt - eye;
    dir = normalize(dir);
    m_eyePos = eye;
    float phi = atan2(dir.x, dir.z);
    float theta = -DirectX::XM_PIDIV2 + acos(dir.y);
    m_phi = 0;
    m_theta = 0;
    Rotate(phi, theta);
}