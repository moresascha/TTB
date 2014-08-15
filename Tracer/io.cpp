#include "io.h"
#include <fstream>
#include <windows.h>
#include <vector>
#include "print.h"
#include "FreeImage.h"
#include <sstream>
#include <assert.h>

extern "C" bool FindFilePath(const char* fileName, std::string& path, std::string* _dir/* = NULL */)
{
    std::string dir;
    if(_dir != NULL)
    {
        dir = *_dir;
    }
    else
    {
        dir = dir + "../"; //root
    }

    WIN32_FIND_DATAA fileData;

    HANDLE fileHandle = FindFirstFileA((dir + fileName).c_str(), &fileData);

    if(fileHandle != INVALID_HANDLE_VALUE)
    {
        path = dir + "/" + fileName;
        FindClose(fileHandle);
        return true;
    }

    fileHandle = FindFirstFileA((dir + "/*").c_str(), &fileData);

    if(INVALID_HANDLE_VALUE == fileHandle)
    {
        return false;
    }

    while(FindNextFileA(fileHandle, &fileData))
    {
        if(fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(fileData.cFileName, "..") > 0)
        {
            //print("DIR REC SEARCH HERE into %s, %s\n", fileData.cFileName, dir.c_str());
            std::string ndir = dir + "/" + fileData.cFileName + "/";
            if(FindFilePath(fileName, path, &ndir))
            {
                FindClose(fileHandle);
                return true;
            }
        }
        /*else
        {
            print(" %s\n", fileData.cFileName);
        }*/
    }

    FindClose(fileHandle);

    return false;
}

extern "C" int ReadTextFile(const char* file, std::string& dst)
{
    std::ifstream s;
    s.open(file, NULL);
    if(s.fail())
    {
        print("File not found %s\n", file);
        return 0;
    }
    std::string line;
    if(s.is_open())
    {
        while(std::getline(s, line))
        {
            dst += line + "\n";
        }
    }
    s.close();
    return 1;
}

TextureData GetTextureData(const char* file, FREE_IMAGE_FORMAT format, int flags)
{
    FIBITMAP* image = FreeImage_Load(format, file, flags);
    //FreeImage_AdjustGamma(image, 2.2);
    int w = FreeImage_GetWidth(image);
    int h = FreeImage_GetHeight(image);
    int type = FreeImage_GetColorType(image);
    int channel = 0;
    switch(type)
    {
    case FIC_MINISBLACK : channel = 1; break;
    case FIC_RGB : channel = 3; break;
    case FIC_RGBALPHA : channel = 4; break;
    default : print("Unknown color type\n"); exit(-1); break;
    }

    if (FreeImage_GetBPP(image) != 32)
    {
        FIBITMAP* old = image;
        image = FreeImage_ConvertTo32Bits(old);
        FreeImage_Unload(old);
    }

    unsigned char* texture = new unsigned char[4 * w * h];
    uint index = 0;
    for(unsigned int y = 0; y < h; ++y)
    {
        for(unsigned int x = 0; x < w; ++x)
        {
            RGBQUAD color;
            memset(&color, 0, sizeof(RGBQUAD));
            FreeImage_GetPixelColor(image, x, y, &color);
            texture[index++] = color.rgbRed;
            texture[index++] = color.rgbGreen;
            texture[index++] = color.rgbBlue;
            texture[index++] = color.rgbReserved;
        }
    }

    FreeImage_Unload(image);

    TextureData td;
    td.data = (uchar4*)texture;
    td.width = w;
    td.height = h;
    td.channel = channel;

    return td;
}

extern "C" TextureData GetTexturePNGData(const char* file)
{
    return GetTextureData(file, FIF_PNG, PNG_DEFAULT);
}

extern "C" TextureData GetTextureJPGData(const char* file)
{
    return GetTextureData(file, FIF_JPEG, JPEG_DEFAULT);
}

int GetMaterial(const char* file, std::map<std::string, RawMaterial>& mats)
{
    std::ifstream s;
    s.open(file, NULL);
    if(s.fail())
    {
        return 0;
    }
    std::string line;

    RawMaterial* currentMat = NULL;

    while(s.good())
    {
        std::string flag;
        s >> flag;
        if(flag == "newmtl")
        {
            std::string name;
            s >> name;
            mats.insert(std::pair<std::string, RawMaterial>(name, RawMaterial()));
            currentMat = &mats[name];
        }
        else if(flag == "Ns")
        {
            s >> currentMat->specularExp;
        }
        else if(flag == "Ka")
        {
            s >> currentMat->ambientI.x;
            s >> currentMat->ambientI.y;
            s >> currentMat->ambientI.z;
        }
        else if(flag == "Kd")
        {
            s >> currentMat->diffuseI.x;
            s >> currentMat->diffuseI.y;
            s >> currentMat->diffuseI.z;
        }
        else if(flag == "Ks")
        {
            s >> currentMat->specularI.x;
            s >> currentMat->specularI.y;
            s >> currentMat->specularI.z;
        }
        else if(flag == "Alpha")
        {
            s >> currentMat->alpha;
        }
        else if(flag == "IOR")
        {
            s >> currentMat->ior;
        }
        else if(flag == "Reflectivity")
        {
            s >> currentMat->reflectivity;
            if(currentMat->reflectivity == 0)
            {
                currentMat->reflectivity = 0.01f;
            }
            //currentMat->reflectivity *= 2;
        }
        else if(flag == "Fresnel_r")
        {
            s >> currentMat->fresnel_r;
        }
        else if(flag == "Fresnel_t")
        {
            s >> currentMat->fresnel_t;
        }
        else if(flag == "Mirror")
        {
            int i;
            s >> i;
            currentMat->mirror = i != 0;
        }
        else if(flag == "map_Kd")
        {
            s >> currentMat->texFile;
        }
    }

    return 1;
}

struct ThreadData
{
    std::string text;
       
    std::vector<Position> positions;
    std::vector<Normal> normals;
    std::vector<TexCoord> texCoords;

    std::vector<int> posindex;
    std::vector<int> nindex;
    std::vector<int> tcindex;

    bool hasTC;
};

template <typename Stream>
int ReadObjFileFromStream(Stream& s, ThreadData* td)
{
    std::string lines[3];
    //std::stringstream ss;
    bool hasTC = td->hasTC;
    while(s.good())
    {
        std::string flag;
        s >> flag;

        if(flag == "v")
        {
            Position v;
            s >> v.x;
            s >> v.y;
            s >> v.z;

            td->positions.push_back(v);
        }
        else if(flag == "vn")
        {
            Normal v;
            s >> v.x;
            s >> v.y;
            s >> v.z;

            td->normals.push_back(v);
        }
        else if(flag == "vt")
        {
            TexCoord v;
            s >> v.x;
            s >> v.y;
            td->texCoords.push_back(v);
        }
        else if(flag == "f")
        {
            lines[0].clear();
            lines[1].clear();
            lines[2].clear();

            s >> lines[0];
            s >> lines[1];
            s >> lines[2];

            std::stringstream ss;
            ss << lines[0] << " " << lines[1] << " " << lines[2];

            //bool hasTC = !contains(lines[0].c_str(), "//");

            int ip0;
            int in0;
            int it0 = 0;

            int ip1;
            int in1;
            int it1 = 0;

            int ip2;
            int in2;
            int it2 = 0;

            char d;

            ss >> ip0; ss >> d;

            if(hasTC)
            {
                ss >> it0;
            }
            
            ss >> d;

            ss >> in0;

            //--

            ss >> ip1; ss >> d;

            if(hasTC)
            {
                ss >> it1;
            }
            
            ss >> d;

            ss >> in1;

            //--

            ss >> ip2; ss >> d;

            if(hasTC)
            {
                ss >> it2;
            }
            
            ss >> d;

            ss >> in2;

            assert(ip2 > 0 && ip1 > 0 && ip0 > 0);
            assert(in2 > 0 && in1 > 0 && in0 > 0);
            
            td->posindex.push_back(ip2 - 1);
            td->posindex.push_back(ip1 - 1);
            td->posindex.push_back(ip0 - 1);

            td->nindex.push_back(in2 - 1);
            td->nindex.push_back(in1 - 1);
            td->nindex.push_back(in0 - 1);

            if(hasTC)
            {
                td->tcindex.push_back(it2 - 1);
                td->tcindex.push_back(it1 - 1);
                td->tcindex.push_back(it0 - 1);
            }
//             else
//             {
//                 td->tcindex.push_back(0);
//                 td->tcindex.push_back(0);
//                 td->tcindex.push_back(0);
//             }
        }
        else
        {
            std::getline(s, flag);
        }
    }
    return 1;
}

unsigned long ThreadProc(void* lpdwThreadParam)
{
    ThreadData* d = (ThreadData*)lpdwThreadParam;
    std::stringstream ss;
    ss << d->text;
    int ok = ReadObjFileFromStream(ss, d);
    return 0;
}

HANDLE ParseObjRange(ThreadData* data)
{
    HANDLE pHandle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)&ThreadProc, (void*)data, 0, NULL);
    if(pHandle == NULL)
    {
        print("Failed to start thread!\n");
        CloseHandle(pHandle);
    }
    return pHandle;
// 
//     ThreadProc(data);
//     return NULL;
}

bool contains(const char* text, const char* pattern)
{
    for(;*text;)
    {
        const char* c = pattern;
        int i;
        for(i = 0; *c; ++i, ++c)
        {
            if(text[i] != *c)
            {
                break;
            }
        }
        if(*c == '\0')
        {
            return true;
        }
        ++text;
    }
    return false;
}

bool beginsWith(const char* text, const char* pattern)
{
    const char* c = pattern;
    for(int i = 0; *c; ++c, ++i)
    {
        if(text[i] != pattern[i])
        {
            return false;
        }
    }
    return true;
}

extern "C" int ReadObjFileThreaded(const char* file, RawTriangles& tries)
{
    print("Loading obj file '%s' ... \n", file);
    std::ifstream s;
    s.open(file, NULL);
    if(s.fail())
    {
        return 0;
    }
    int lines = 0;
    std::string text;

    int rangeLength = 50000; // lines per thread

    std::vector<ThreadData*> data;
    std::vector<HANDLE> handles;
    std::string currentMat;
    int start = 0;
    int end = 0;
    bool firstIval = true;
    bool hasTC = false;
    int numThreads = 0;
    do
    {
        std::string flag;
        std::getline(s, flag);

        if(beginsWith(flag.c_str(), "mtllib"))
        {
            std::string mtlFile(flag.begin() + 7, flag.end());
            std::string matFile;
            FindFilePath(mtlFile.c_str(), matFile);
            GetMaterial(matFile.c_str(), tries.materials);
            continue;
        }
        else if(beginsWith(flag.c_str(), "usemtl"))
        {
            if(!firstIval)
            {
                tries.intervals.push_back(IndexBufferInterval(start, end, currentMat));
            }
            currentMat = std::string(flag.begin() + 7, flag.end());
            firstIval = false;
            start = end;
            continue;
        }
        else if(beginsWith(flag.c_str(), "vt"))
        {
            hasTC = true;
        }
        else if(beginsWith(flag.c_str(), "f"))
        {
            end++;
        }

        text += flag + "\n";
        lines++;

        if((lines-1) && (rangeLength == lines) || (!s.good() && text.size() > 0))
        {
            ThreadData* td = new ThreadData();
            td->hasTC = hasTC;
            data.push_back(td);
            td->text = text;
            handles.push_back(ParseObjRange(td));
            lines = 0;
            text = "";
            numThreads++;
        }
    }
    while(s.good());

    print("    Started %d Thread(s)\n", numThreads);

    tries.intervals.push_back(IndexBufferInterval(start, end, currentMat));

    s.close();

    WaitForMultipleObjects(handles.size(), &handles[0], true, INFINITE);

    ThreadData d;
    for(int i = 0; i < data.size(); ++i)
    {
        d.normals.insert(d.normals.end(), data[i]->normals.begin(), data[i]->normals.end());
        d.texCoords.insert(d.texCoords.end(), data[i]->texCoords.begin(), data[i]->texCoords.end());
        d.positions.insert(d.positions.end(), data[i]->positions.begin(), data[i]->positions.end());

        for(int j = 0; j < data[i]->posindex.size(); ++j)
        {
            tries.positions.push_back(d.positions[data[i]->posindex[j]]);
            tries.normals.push_back(d.normals[data[i]->nindex[j]]);
        }

        for(int j = 0; j < data[i]->tcindex.size(); ++j)
        {
            uint dd = data[i]->tcindex[j];
            tries.tcoords.push_back(d.texCoords[dd == -1 ? 0 : dd]);
        }

        delete data[i];
    }

    print("    done.\n");

    return 1;
}


extern "C" int ReadObjFile(const char* file, RawTriangles& tries)
{
    std::ifstream s;
    s.open(file, NULL);
    if(s.fail())
    {
        return 0;
    }

    std::vector<Position> positions;
    std::vector<Normal> normals;
    std::vector<TexCoord> texCoords;

    std::string currentMat;
    int start = 0;
    int end = 0;

    bool firstIval = true;

    while(s.good())
    {
        std::string flag;
        s >> flag;
        if(flag == "mtllib")
        {
            std::string mtlFile;
            s >> mtlFile;
            std::string matFile;
            FindFilePath(mtlFile.c_str(), matFile);
            GetMaterial(matFile.c_str(), tries.materials);
        }
        else if(flag == "v")
        {
            Position v;
            s >> v.x;
            s >> v.y;
            s >> v.z;

            positions.push_back(v);
        }
        else if(flag == "vn")
        {
            Normal v;
            s >> v.x;
            s >> v.y;
            s >> v.z;

            normals.push_back(v);
        }
        else if(flag == "vt")
        {
            TexCoord v;
            s >> v.x;
            s >> v.y;

            texCoords.push_back(v);
        }
        else if(flag == "f")
        {
            //std::getline(s, flag, '\n');

            end++;

            int ip0;
            int in0;
            int it0;

            int ip1;
            int in1;
            int it1;

            int ip2;
            int in2;
            int it2;

            char d;

            s >> ip0; s >> d;

            s >> it0; s >> d;

            s >> in0;
            //--
            s >> ip1; s >> d;

            s >> it1; s >> d;

            s >> in1;
            //--
            s >> ip2; s >> d;

            s >> it2; s >> d;

            s >> in2;

            Position& p0 = positions[ip0 - 1];
            Normal& n0 = normals[in0 - 1];
            TexCoord& t0 = texCoords[it0 - 1];
           
            Position& p1 = positions[ip1 - 1];
            Normal& n1 = normals[in1 - 1];
            TexCoord& t1 = texCoords[it1 - 1];

            Position& p2 = positions[ip2 - 1];
            Normal& n2 = normals[in2 - 1];
            TexCoord& t2 = texCoords[it2 - 1];

            tries.positions.push_back(p2);
            tries.positions.push_back(p1);
            tries.positions.push_back(p0);

            tries.normals.push_back(n2);
            tries.normals.push_back(n1);
            tries.normals.push_back(n0);

            tries.tcoords.push_back(t2);
            tries.tcoords.push_back(t1);
            tries.tcoords.push_back(t0);
        }
        else if(flag == "usemtl")
        {
            if(!firstIval)
            {
                tries.intervals.push_back(IndexBufferInterval(start, end, currentMat));
            }
            firstIval = false;
            start = end;
            s >> currentMat;
        }
        else
        {
            std::getline(s, flag);
        }
    }

    tries.intervals.push_back(IndexBufferInterval(start, end, currentMat));

    s.close();

    return 1;
}