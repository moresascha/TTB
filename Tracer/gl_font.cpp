#include "io.h"
#include "gl_layer.h"
#include "gl_globals.h"
#include "print.h"
#include <sstream>

glProgram* g_fontProg;
glGeometry* g_fontGeo;
GLuint g_fontTexture;

struct CMFontMetrics
{
    float leftU;
    float rightU; 
    int pixelWidth;
};

struct CMCharMetric
{
    unsigned char id;
    int x;
    int y;
    int width;
    int height;
    int xoffset;
    int yoffset;
    int xadvance;
};

struct CMFontStyle
{
    bool italic;
    bool bold;
    int charCount;
    int lineHeight;
    int texWidth;
    int texHeight;
    int size;
    int base;
    std::string textureFile;
    std::string metricFile;
    std::string name;
};

std::map<unsigned char, CMCharMetric> m_metrics;
CMFontStyle m_style;

namespace util
{
    std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) 
    {
        std::stringstream ss(s);
        std::string item;
        while(std::getline(ss, item, delim)) 
        {
            elems.push_back(item);
        }
        return elems;
    }

    std::vector<std::string> split(const std::string &s, char delim)
    {
        std::vector<std::string> elems;
        return split(s, delim, elems);
    }
}

bool ReadBMFont(const char* file)
{
    std::string path;
    FindFilePath(file, path);

    std::ifstream metricsStream(path);
    if(!metricsStream.good())
    {
        return false;
    }

    while(metricsStream.good())
    {
        std::string s;
        std::getline(metricsStream, s);
        std::vector<std::string> ss = util::split(s, ' ');
        if(ss.size() == 0) 
        {
            continue;
        }
        if(ss[0].compare("info") == 0)
        {
            m_style.size = atoi((util::split(ss[2], '=')[1]).c_str());
            m_style.name = util::split(ss[1], '=')[1];
            m_style.bold = atoi((util::split(ss[3], '=')[1]).c_str()) != 0;
            m_style.italic = atoi((util::split(ss[4], '=')[1]).c_str()) != 0;
        }
        else if(ss[0].compare("common") == 0)
        {   
            m_style.lineHeight = atoi((util::split(ss[1], '=')[1]).c_str());
            m_style.base = atoi((util::split(ss[2], '=')[1]).c_str());
            m_style.texWidth = atoi((util::split(ss[3], '=')[1]).c_str());
            m_style.texHeight = atoi((util::split(ss[4], '=')[1]).c_str());
        }
        else if(ss[0].compare("page") == 0)
        {
            m_style.textureFile = util::split(ss[2], '=')[1];
        }
        else if(ss[0].compare("chars") == 0)
        {
            m_style.charCount = atoi((util::split(ss[1], '=')[1]).c_str());
        }
        else if(ss[0].compare("char") == 0)
        {
            std::vector<std::string> tokens;
            for(uint i  = 1; i < ss.size(); ++i)
            {
                if(ss[i].compare(""))
                {
                    std::vector<std::string> split = util::split(ss[i], '=');
                    tokens.push_back(split[1]);
                }
            }
            UCHAR c = (UCHAR)atoi(tokens[0].c_str());
            m_metrics[c].id = c;
            m_metrics[c].x = atoi(tokens[1].c_str());
            m_metrics[c].y = atoi(tokens[2].c_str());
            m_metrics[c].width = atoi(tokens[3].c_str());
            m_metrics[c].height = atoi(tokens[4].c_str());
            m_metrics[c].xoffset = atoi(tokens[5].c_str());
            m_metrics[c].yoffset = atoi(tokens[6].c_str());
            m_metrics[c].xadvance = atoi(tokens[7].c_str());
        }
    }
    metricsStream.close();

    return true;
}

bool FontInit(void)
{
    g_fontProg = g_fontProg->CreateProgramFromFile("glsl/font_vs.glsl", "glsl/font_fs.glsl");

    if(g_fontProg == NULL)
    {
        return false;
    }

    g_fontGeo = new glGeometry();

    g_fontGeo->Init(20 * sizeof(float));

    if(!ReadBMFont("font_16.fnt"))
    {
        return false;
    }

    std::string path;

    FindFilePath(m_style.textureFile.c_str(), path);

    TextureData td = GetTexturePNGData(path.c_str());

    g_fontProg->Bind();
    glGenTextures(1, &g_fontTexture);
    glActiveTexture(GL_TEXTURE0 + TEXTURE_SLOT_FONT);
    glBindTexture(GL_TEXTURE_2D, g_fontTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, td.width, td.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, td.data);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUniform1i(glGetUniformLocation(g_fontProg->Id(), "tex"), TEXTURE_SLOT_FONT);

    delete[] td.data;

    return true;
}

void FontBeginDraw(void)
{
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    g_fontProg->Bind();
    g_fontGeo->Bind();
}

void FontEndDraw(void)
{
    glDisable(GL_BLEND);
}

void FontDrawBMFont(const char* text, float x, float y)
{
    if(x < 0 || x > 1 || y < 0 || y > 1)
    {
        return;
    }

    int curserX = 0;
    int curserY = 0;

//     RECT r;
//     GetClientRect(GetWindowHandle(), &r); 

    float w = (float) 768;
    float h = (float) 768;

    for(const char* cp = text; *cp != '\0'; ++cp)
    {
        char c = *cp;
        if(c == '\n')
        {
            curserY -= m_style.lineHeight;
            curserX = 0;
            continue;
        }

        auto it = m_metrics.find(c);
        if(it == m_metrics.end())
        {
            continue;
        }
        const CMCharMetric* metric = &it->second;

        float u0 = metric->x / (float)m_style.texWidth;
        float v0 = 1 - metric->y/ (float)m_style.texHeight;
        float u1 = (metric->x + metric->width ) / (float)m_style.texWidth;
        float v1 = 1 - (metric->y + metric->height) / (float)m_style.texHeight;

        int quadPosX = (int)(x * w);
        int quadPosY = (int)((1-y) * h);

        quadPosX += curserX + metric->xoffset;
        quadPosY -= metric->yoffset - curserY;

        float nposx = 2.0f * quadPosX / w - 1;
        float nposy = 2.0f * quadPosY / h - 1;

        float nposx1 = 2.0f * (quadPosX + metric->width) / w - 1;
        float nposy1 = 2.0f * (quadPosY - metric->height) / h - 1;

        float localVertices[20] = 
        {
            nposx,  nposy1, 0, u0, v1,
            nposx1, nposy1, 0, u1, v1,
            nposx,  nposy, 0, u0, v0,
            nposx1, nposy, 0, u1, v0
        };

        g_fontGeo->UploadData(localVertices);
        g_fontGeo->Draw();

        curserX += metric->xadvance;
    }
}

void FontDrawText(const char* text, float px, float py) /*x & y in [0,1]*/
{
    FontDrawBMFont(text, px, py);
    return;
    //todo
    const int window_w = 768;
    const int window_h = 768;
    const int gw = 16;
    const int gh = 16;
    const int texSize = 512;

    float tw = 2*gw / (float)texSize;
    float th = 2*gh / (float)texSize;
    float w = 4*gw / (float)window_w;
    float h = 4*gh / (float)window_h;

    float xoff = 0;
    float yoff = 0;

    for(const char* cp = text; *cp != '\0'; ++cp)
    {
        char c = *cp;

        if(c == '\n')
        {
            yoff += 2*(gh+gh/2) / (float)window_h;
            xoff = 0;
            continue;
        }

        float tx = (((int)c) % gw) / (float)gw;
        float ty = 1 - (((int)c) / gh) / (float)gh - th;

        float x = -1 + 2 * px + xoff;
        float y = 1 - 2 * py - h - yoff;

        float verts[20] = 
        {
            x    ,     y, 0, tx     , ty     ,
            x + w, y    , 0, tx + tw, ty     ,
            x    , y + h, 0, tx     , ty + th,
            x + w, y + h, 0, tx + tw, ty + th
        };

        g_fontGeo->UploadData(verts);
        g_fontGeo->Draw();
        xoff += 2*gw / (float)window_w;
    }
}

void FontDestroy(void)
{
    glDeleteTextures(1, &g_fontTexture);
    delete g_fontProg;
    delete g_fontGeo;
}