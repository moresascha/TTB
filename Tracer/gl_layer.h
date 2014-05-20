#pragma once
#include <gl/glew.h>
#include <gl/freeglut.h>

HGLRC CreateGLContextAndMakeCurrent(HWND hWnd);

void ReleaseGLContext(HGLRC hRC);

HWND GetWindowHandle(void);

class glProgram
{
    GLuint m_program;
    GLuint m_vertexShader;
    GLuint m_fragmentShader;
public:
    glProgram(void);

    void Init(void);

    void SetVertexShader(const char* source);

    void SetFragmentShader(const char* source);

    int Compile(void);

    void Bind(void);

    GLuint Id(void) {return m_program; }

    static glProgram* CreateProgramFromFile(const char* vs, const char* fs);

    ~glProgram(void);
};

class glGeometry
{
    GLuint m_vertexBuffer;
    GLuint m_indexBuffer;
    GLuint m_vertexArray;

    int m_vertexBytes;
    int m_indexBytes;

    GLenum m_topo;

public:
    glGeometry(void);

    void Init(int vertexBytes, int intBytes = 0);

    void UploadData(float* vertexData, int* indices = NULL);

    void Bind(void);

    void Delete(void);

    void SetTopo(GLenum topo)
    {
        m_topo = topo;
    }

    void Draw(void);

    ~glGeometry(void);
};

class glTextureBuffer
{
    GLuint m_texture;
    GLuint m_buffer;

    void _DeleteBuffer(void);

public:
    glTextureBuffer(void);

    void Resize(GLuint newSize);

    GLuint BufferId();

    GLuint TextureId();

    void BindToTextureSlot(int slot);

    ~glTextureBuffer(void);
};