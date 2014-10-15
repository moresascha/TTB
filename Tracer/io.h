#pragma once
#include <string>
#include <vector>
#include <map>
#include "cuda/types.cuh"

struct TextureData
{
    uchar4* data;
    int channel;
    int width;
    int height;
};

struct RawMaterial
{
    float3 diffuseI;
    float3 ambientI;
    float3 specularI;
    float specularExp;

    bool mirror;
    float alpha;
    float fresnel_r;
    float fresnel_t;
    float ior;
    float reflectivity;

    std::string texFile;
};

struct IndexBufferInterval
{
    IndexBufferInterval(void) { }

    IndexBufferInterval(int s, int e, std::string& mat)
    {
        material = mat;
        start = s;
        end  = e;
    }
    int start;
    int end;
    std::string material;
};

struct RawTriangles
{
    std::vector<Position> positions;
    std::vector<Normal> normals;
    std::vector<TexCoord> tcoords;

    std::map<std::string, RawMaterial> materials;

    std::vector<IndexBufferInterval> intervals;


    byte GetMaterialIndex(const std::string& name)
    {
        byte index = 0;
        if(materials.size() > 256)
        {
            throw "Error";
        }
        for(auto it = materials.begin(); it != materials.end(); ++it)
        {
            if(it->first == name)
            {
                return index;
            }
            index++;
        }
        return -1;
    }
};

void Serialize(const RawTriangles* data, const char* fileName);

void DeSerialize(RawTriangles* data, const char* fileName);

extern "C" bool FindFilePath(const char* fileName, std::string& path, std::string* dir = NULL);

extern "C" TextureData GetTextureJPGData(const char* file);

extern "C" TextureData GetTexturePNGData(const char* file);

extern "C" TextureData GetTextureTGAData(const char* file);

extern "C" int ReadTextFile(const char* file, std::string& dst);

extern "C" void WritePNGFile(float4* data, uint w, uint h, const char* fileName);

extern "C" int ReadObjFile(const char* file, RawTriangles& tries);

extern "C" int ReadObjFileThreaded(const char* file, RawTriangles& tries);