#include "gl_layer.h"
#include <string>
#include "io.h"
#include <sstream>
#include "print.h"

glProgram::glProgram(void) : m_program(0), m_fragmentShader(0), m_vertexShader(0)
{

}

void glProgram::Init(void)
{
    m_program = glCreateProgram();
}

int compileShader(GLuint shader)
{
    glCompileShader(shader);

    char log[2048];
    int size = 0;
    glGetShaderInfoLog(shader, 2048, &size, log);
    if(size)
    {
        std::stringstream ss;
        ss << log;
        OutputDebugStringA(ss.str().c_str());
        OutputDebugStringA("\n");
        return 0;
    }

    return 1;
}

int glProgram::Compile(void)
{
    if(!compileShader(m_vertexShader))
    {
        return 0;
    }
    
    if(!compileShader(m_fragmentShader))
    {
        return 0;
    }
    
    glLinkProgram(m_program);

    char log[2048];
    int size = 0;
    glGetProgramInfoLog(m_program, 2048, &size, log);

    if(size)
    {
        std::stringstream ss;
        ss << log;
        OutputDebugStringA(ss.str().c_str());
        OutputDebugStringA("\n");
        return 0;
    }

    glValidateProgram(m_program);

    return 1;
}

void glProgram::SetFragmentShader(const char* source)
{
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(m_fragmentShader, 1, &source, 0);
    glAttachShader(m_program, m_fragmentShader);
}

void glProgram::SetVertexShader(const char* source)
{
    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_vertexShader, 1, &source, 0);
    glAttachShader(m_program, m_vertexShader);
}

glProgram* glProgram::CreateProgramFromFile(const char* vs, const char* fs)
{
    glProgram* prog = new glProgram();
    prog->Init();

    std::string src;
    if(!ReadTextFile(vs, src))
    {
        return NULL;
    }

    prog->SetVertexShader(src.c_str());

    src = "";
    if(!ReadTextFile(fs, src))
    {
        return NULL;
    }

    prog->SetFragmentShader(src.c_str());

    if(prog->Compile())
    {
        return prog;
    }

    delete prog;
    return NULL;
}

void glProgram::Bind(void)
{
    glUseProgram(m_program);
}

glProgram::~glProgram(void)
{
    if(m_vertexShader)
    {
        glDeleteShader(m_vertexShader);
    }

    if(m_fragmentShader)
    {
        glDeleteShader(m_fragmentShader);
    }

    if(m_program)
    {
        glDeleteProgram(m_program);
    }
}

glGeometry::glGeometry(void) : m_indexBuffer(0), m_vertexBuffer(0), m_vertexArray(0), m_indexBytes(0), m_vertexBytes(0), m_topo(GL_TRIANGLE_STRIP)
{
    
}

void glGeometry::Init(int vertexBytes, int intBytes /* = 0 */)
{
    m_indexBytes = intBytes;
    m_vertexBytes = vertexBytes;

    glGenVertexArrays(1, &m_vertexArray);
    glBindVertexArray(m_vertexArray);

    glGenBuffers(1, &m_vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertexBytes, NULL, GL_STATIC_DRAW);
    if(intBytes)
    {
        glGenBuffers(1, &m_indexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, intBytes, NULL, GL_STATIC_DRAW);
    }

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(float) * 5, 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, false, sizeof(float) * 5, (GLvoid*)(sizeof(float) * 3));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void glGeometry::UploadData(float* vertexData, int* indices /* = NULL */)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_vertexBytes, vertexData);

    if(indices && m_indexBuffer)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, m_indexBytes, indices);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void glGeometry::Bind(void)
{
    glBindVertexArray(m_vertexArray);
}

void glGeometry::Draw(void)
{
    Bind();
    if(m_indexBuffer)
    {
        glDrawElements(m_topo, m_indexBytes / sizeof(int), GL_UNSIGNED_INT, NULL);
    }
    else
    {
        glDrawArrays(m_topo, 0, m_vertexBytes / (5 * sizeof(float)));
    }
}

glGeometry::~glGeometry(void)
{
    if(m_indexBuffer)
    {
        glDeleteBuffers(1, &m_indexBuffer);
    }

    if(m_vertexBuffer)
    {
        glDeleteBuffers(1, &m_vertexBuffer);
    }

    if(m_vertexArray)
    {
        glDeleteVertexArrays(1, &m_vertexArray);
    }
}

glTextureBuffer::glTextureBuffer(void) : m_texture(0), m_buffer(0)
{
    
}

void glTextureBuffer::_DeleteBuffer(void)
{
    if(m_buffer)
    {
        glDeleteBuffers(1, &m_buffer);
    }
}

void glTextureBuffer::Resize(GLuint newSize)
{
    if(!m_texture)
    {
        glGenTextures(1, &m_texture);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, m_texture);

    _DeleteBuffer();
    glGenBuffers(1, &m_buffer);
    glBindBuffer(GL_TEXTURE_BUFFER, m_buffer);
    glBufferData(GL_TEXTURE_BUFFER, newSize, NULL, GL_DYNAMIC_DRAW);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, m_buffer);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

GLuint glTextureBuffer::BufferId()
{
    return m_buffer;
}

GLuint glTextureBuffer::TextureId()
{
    return m_texture;
}

void glTextureBuffer::BindToTextureSlot(int slot)
{
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_BUFFER, m_texture);
    glActiveTexture(GL_TEXTURE0);
}

glTextureBuffer::~glTextureBuffer(void)
{
    if(m_texture)
    {
        glDeleteTextures(1, &m_texture);
    }
    _DeleteBuffer();
}

HGLRC CreateGLContextAndMakeCurrent(HWND hWnd)
{
    HDC hDC;
    if(!(hDC = GetDC(hWnd)))
    {
        return 0;
    }

    int colorBits = 32;
    PIXELFORMATDESCRIPTOR pfd =                  // pfd Tells Windows How We Want Things To Be
    {
        sizeof(PIXELFORMATDESCRIPTOR),                  // Size Of This Pixel Format Descriptor
        1,                              // Version Number
        PFD_DRAW_TO_WINDOW |                        // Format Must Support Window
        PFD_SUPPORT_OPENGL |                        // Format Must Support OpenGL
        PFD_DOUBLEBUFFER,                       // Must Support Double Buffering
        PFD_TYPE_RGBA,                          // Request An RGBA Format
        colorBits,                               // Select Our Color Depth
        0, 0, 0, 0, 0, 0,                       // Color Bits Ignored
        0,                              // No Alpha Buffer
        0,                              // Shift Bit Ignored
        0,                              // No Accumulation Buffer
        0, 0, 0, 0,                         // Accumulation Bits Ignored
        16,                             // 16Bit Z-Buffer (Depth Buffer)
        0,                              // No Stencil Buffer
        0,                              // No Auxiliary Buffer
        PFD_MAIN_PLANE,                         // Main Drawing Layer
        0,                              // Reserved
        0, 0, 0                             // Layer Masks Ignored
    };

    int pf = 0;

    pf = ChoosePixelFormat(hDC, &pfd);

    if(!pf)
    {
        return 0;
    }

    if(!SetPixelFormat(hDC, pf, &pfd))
    {
        return 0;
    }

    HGLRC hRC;
    if(!(hRC = wglCreateContext(hDC)))
    {
        return 0;
    }

    if(!wglMakeCurrent(hDC, hRC))
    {
        wglDeleteContext(hRC);
        return 0;
    }

    if(glewInit() != GLEW_OK )
    {
        wglDeleteContext(hRC);
        return 0;
    }

    return hRC;
}

void ReleaseGLContext(HGLRC hRC)
{
    if(wglMakeCurrent(NULL, NULL))
    {
        wglDeleteContext(hRC);
    }
}