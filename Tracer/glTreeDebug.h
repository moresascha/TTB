#pragma once
#include <ct_runtime.h>
#include "ct_debug.h"
#include "gl_layer.h"

struct Line
{
    CTreal3 start;
    CTreal3 end;
};

class glDebugLayer : public ICTTreeDebugLayer
{
private:
    std::vector<Line> m_lines;
    glGeometry m_line;
    glGeometry m_box;
    glProgram* m_pProg;

    bool m_glGeoInit;

public:
    glDebugLayer(void)
    {
        m_glGeoInit = false;
        m_line.SetTopo(GL_LINES);
        m_line.Init(10 * sizeof(float));
        m_pProg = glProgram::CreateProgramFromFile("glsl/vs_p.glsl", "glsl/fs_p.glsl");
    }

    void SetDrawColor(float r, float g, float b)
    {
        GLuint loc = glGetUniformLocation(GetProgram()->Id(), "color");
        float c[3] = {r, g, b};
        glUniform3fv(loc, 1, c);
    }

    void SetDrawColor(const CTreal3& c)
    {
        SetDrawColor(c.x, c.y, c.z);
    }

    void BeginDraw(void)
    {
        m_pProg->Bind();
        SetDrawColor(0.025f,0.1f,0.025f);
        srand(0);
    }

    void ResetGeometry(void)
    {
        m_lines.clear();
        m_line.Delete();
        m_glGeoInit = false;
    }

    glProgram* GetProgram(void)
    {
        return m_pProg;
    }

    void EndDraw(void)
    {
        glUseProgram(0);
    }

    void DrawLine(const CTreal3& start, const CTreal3& end)
    {
        Line l;
        l.start = start; 
        l.end = end;
        m_lines.push_back(l);
        
//         float vertex[10];
//         vertex[0] = start.x;
//         vertex[1] = start.y;
//         vertex[2] = start.z;
//         vertex[3] = 0;
//         vertex[4] = 0;
// 
//         vertex[5] = end.x;
//         vertex[6] = end.y;
//         vertex[7] = end.z;
//         vertex[8] = 1;
//         vertex[9] = 1;
// 
//         m_line.UploadData(vertex);
// 
//         m_line.Draw();
    }

    void DrawGLGeo(void)
    {
        if(!m_glGeoInit)
        {
            int bytes = 10 * sizeof(float) * (int)m_lines.size();

            m_line.Init(bytes);

            float* vertexData = new float[10 * m_lines.size()];

            for(int i = 0; i < m_lines.size(); ++i)
            {
                Line l = m_lines[i];
                vertexData[10 * i + 0] = l.start.x;
                vertexData[10 * i + 1] = l.start.y;
                vertexData[10 * i + 2] = l.start.z;
                vertexData[10 * i + 3] = 0;
                vertexData[10 * i + 4] = 0;

                vertexData[10 * i + 5] = l.end.x;
                vertexData[10 * i + 6] = l.end.y;
                vertexData[10 * i + 7] = l.end.z;
                vertexData[10 * i + 8] = 1;
                vertexData[10 * i + 9] = 1;
            }

            m_line.UploadData((float*)vertexData);
            delete[] vertexData;

            m_glGeoInit = true;
        }

        m_line.Draw();
    }

    void _DrawBox(const ICTAABB& aabb)
    {
        float3 mini = aabb.GetMin();
        float3 maxi = aabb.GetMax();

        float d = 0;//1e-2f * (rand() / (float)RAND_MAX);

        mini.x += d;
        mini.y += d;
        mini.z += d;

        maxi.x += d;
        maxi.y += d;
        maxi.z += d;

        //buttom
        DrawLine(mini, make_float3(mini.x, mini.y, maxi.z));
        DrawLine(make_float3(mini.x, mini.y, maxi.z), make_float3(maxi.x, mini.y, maxi.z));
        DrawLine(make_float3(maxi.x, mini.y, maxi.z), make_float3(maxi.x, mini.y, mini.z));
        DrawLine(make_float3(maxi.x, mini.y, mini.z), mini);

        //middle
        DrawLine(mini, make_float3(mini.x, maxi.y, mini.z));
        DrawLine(make_float3(mini.x, mini.y, maxi.z), make_float3(mini.x, maxi.y, maxi.z));
        DrawLine(make_float3(maxi.x, mini.y, maxi.z), make_float3(maxi.x, maxi.y, maxi.z));
        DrawLine(make_float3(maxi.x, mini.y, mini.z), make_float3(maxi.x, maxi.y, mini.z));

        //top
        DrawLine(make_float3(mini.x, maxi.y, mini.z), make_float3(mini.x, maxi.y, maxi.z));
        DrawLine(make_float3(mini.x, maxi.y, maxi.z), make_float3(maxi.x, maxi.y, maxi.z));
        DrawLine(make_float3(maxi.x, maxi.y, maxi.z), make_float3(maxi.x, maxi.y, mini.z));
        DrawLine(make_float3(maxi.x, maxi.y, mini.z), make_float3(mini.x, maxi.y, mini.z));
    }

    void DrawBox(const ICTAABB& aabb)
    {
        //SetDrawColor(1,0,0);
        _DrawBox(aabb);
    }

    void DrawWiredBox(const ICTAABB& aabb)
    {
        //SetDrawColor(1,1,1);
        _DrawBox(aabb);
    }

    ~glDebugLayer(void) 
    {
        if(m_pProg)
        {
             delete m_pProg;
             m_pProg = NULL;
        }
    }
};