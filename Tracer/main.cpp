#include <windows.h>
#include <windowsx.h>
#include "window.h"
#include "gl_layer.h"
#include <cuda_gl_interop.h>
#include <stdlib.h>
#include <tchar.h>
#include <sstream>
#include "print.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include "cuda/tracer_api.cuh"
#include "camera.h"
#include <vector>
#include "io.h"
#include <time.h>
#include <queue>
#include "texture_array.h"
#include "gl_font.h"
#include "gl_globals.h"

#ifdef _DEBUG
#define NUTTY_DEBUG
#endif
#include <Nutty.h>

#include <Copy.h>
#include <DeviceBuffer.h>
#include <cuda/cuda_helper.h>

#include <chimera/api/ChimeraAPI.h>
#include <chimera/Vec4.h>
#include <chimera/Timer.h>
#include <chimera/Vec3.h>
#include <chimera/Mat4.h>
#include <chimera/util.h>

#include "ct_runtime.h"

#include "glTreeDebug.h"
#include "input.h"

#include "cuda/tracer_api.cuh"

#pragma comment(lib, "cuda.lib")

#pragma comment(lib, "glew32.lib")

#pragma comment(lib, "freeimage.lib")

#ifdef _DEBUG
#pragma comment(lib, "Nuttyx64Debug.lib")
#pragma comment(lib, "TreesDebug.lib")
#else
#pragma comment(lib, "Nuttyx64Release.lib")
#pragma comment(lib, "TreesRelease.lib")
#endif

Camera g_cam;
cuTextureAtlas* atlas;
glDebugLayer* debugLayer = NULL;

int width;
int height;

void Resize(int width, int height)
{
    height = height == 0 ? 1 : height;
    width = width == 0 ? 1 : width;
    glViewport(0, 0, width, height);
    g_cam.ComputeProj(width / 2, height);
    if(debugLayer)
    {
        debugLayer->GetProgram()->Bind();
        GLuint loc = glGetUniformLocation(debugLayer->GetProgram()->Id(), "perspective");
        glUniformMatrix4fv(loc, 1, false, (float*)g_cam.GetProjection());
    }
}

WPARAM g_key;
bool g_isKeyDown = false;
bool g_animateCamera = false;
bool g_animateLight = !true;

bool computeMovement(float dt)
{
    if(!g_isKeyDown)
    {
        return false;
    }

    float delta = 4*dt;
    if(g_key == KEY_W)
    {
        g_cam.Move(0,0,delta);
    }
    else if(g_key == KEY_S)
    {
        g_cam.Move(0,0,-delta);
    }
    else if(g_key == KEY_D)
    {
        g_cam.Move(delta,0,0);
    }
    else if(g_key == KEY_A)
    {
        g_cam.Move(-delta,0,0);
    }
    else if(g_key == KEY_C)
    {
        g_cam.Move(0,-delta,0);
    }
    else if(g_key == KEY_V)
    {
        g_cam.Move(0,delta,0);
    }

    return true;
}

Material* g_matToPrint;

void drawUI(void)
{
    std::stringstream ss;
    ss.str("");
    ss << "Tracedepth (-j, +k) = " << RT_GetRecDepth() << "\n";
    ss << "Animate Camera (g) = " << g_animateCamera << "\n";
    ss << "Animate Light (h) = " << g_animateLight << "\n";
    FontBeginDraw();
    FontDrawText(ss.str().c_str(), 0.0025, 0.3);
    FontEndDraw();
}

void computeMaterialChange(void)
{
    //97 ->
//     Material mat;// = cpuMats[0];
// 
    if(g_key == KEY_G)
    {
        g_animateCamera ^= 1;
    }
    else if(g_key == KEY_H)
    {
        g_animateLight ^= 1;
    }
//     else if(g_key == 97)
//     {
//         mat._mirror ^= 1;
//     }
// 
    if(g_key == KEY_K)
    {
        RT_IncDepth();
    }
    else if(g_key == KEY_J)
    {
        RT_DecDepth();
    }
// 
//     //Alpha
//     else if(g_key == 98)
//     {
//         mat._alpha -= 0.01f;
//     }
//     else if(g_key == 99)
//     {
//         mat._alpha += 0.01f;
//     }
// 
//     //Reflectance
//     else if(g_key == 100)
//     {
//         mat._reflectivity -= 0.01f;
//     }
//     else if(g_key == 101)
//     {
//         mat._reflectivity += 0.01f;
//     }
// 
//     //Fresnel_R
//     else if(g_key == 102)
//     {
//         mat._fresnel_r -= 0.01f;
//     }
//     else if(g_key == 103)
//     {
//         mat._fresnel_r += 0.01f;
//     }
// 
//     //Fresnel_T
//     else if(g_key == 104)
//     {
//         mat._fresnel_t -= 0.01f;
//     }
//     else if(g_key == 105)
//     {
//         mat._fresnel_t += 0.01f;
//     }
// 
//     //IOR
//     else if(g_key == KEY_P)
//     {
//         mat._reflectionIndex -= 0.01f;
//     }
//     else if(g_key == KEY_O)
//     {
//         mat._reflectionIndex += 0.01f;
//     }
    
    //CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuMats, (void*)&cpuMats[0], sizeof(Material), cudaMemcpyHostToDevice));
}

unsigned int g_lastX = 0;
unsigned int g_lastY = 0;

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_PAINT:
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_SIZE:
        Resize(LOWORD(lParam),HIWORD(lParam));
        break;
    case WM_ACTIVATE:
        break;
    case WM_SYSCOMMAND:
        {
            switch (wParam)
            {
            case SC_SCREENSAVE:
            case SC_MONITORPOWER:
                return 0;
            }
        } break;
    case WM_MOUSEMOVE:
        {
            if(wParam == MK_LBUTTON)
            {
                int x = GET_X_LPARAM(lParam); 
                int y = GET_Y_LPARAM(lParam);

                int dx = x - g_lastX;
                int dy = y - g_lastY;

                g_cam.Rotate(dx * 1e-3f, dy * 1e-3f);

                g_lastX = x;
                g_lastY = y;
            }
        } break;
    case WM_LBUTTONDOWN:
        {
            g_lastX = GET_X_LPARAM(lParam); 
            g_lastY = GET_Y_LPARAM(lParam);
        } break;
    case WM_KEYUP:
        {
            g_isKeyDown = false;
            return 0;
        }
    case WM_KEYDOWN :
        {
            g_isKeyDown = true;
            g_key = wParam;
            computeMaterialChange();
            return 0;
        } break;
    default:
        break;
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

#define MY_CLASS L"MyClass"

float cubeRand(float s)
{
    return -s + s * 2 * rand() / (float)RAND_MAX;
}

float cubeRand(void)
{
    return cubeRand(1);
}

void checkCudaError(cudaError_t error)
{
    if(error != cudaSuccess)
    {
        print("ERROR: %d\n", error);
        __debugbreak();
    }
}

void checkGLError(void)
{
    int glError = glGetError();
    if(glError != GL_NO_ERROR)
    {
        print("%d\n", glError);
    }
}

struct NodeGPUDataTransformer
{
    nutty::DeviceBuffer<uint> nodeIndexToLeafIndex;
    nutty::DeviceBuffer<uint> lineartreeContent;
    nutty::DeviceBuffer<uint> lineartreeContentCount;
    nutty::DeviceBuffer<uint> lineartreeContentStart;
    nutty::DeviceBuffer<uint> lineartreeLeftNode;
    nutty::DeviceBuffer<uint> lineartreeRightNode;

    nutty::DeviceBuffer<float> lineartreeSplit;
    nutty::DeviceBuffer<byte> lineartreeSplitAxis;
    nutty::DeviceBuffer<byte> lineartreeNodeIsLeaf;

    CTuint leafIndex;
    CTuint startPos;

    NodeGPUDataTransformer(void) : leafIndex(0), startPos(0)
    {

    }

    void Reset(void)
    {
        leafIndex = 0;
        startPos = 0;
    }

    template <typename T>
    void Fill(ICTTree* tree, CT_LINEAR_MEMORY_TYPE type, nutty::DeviceBuffer<T>& buffer)
    {
        uint cnt;
        const void* memory;
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, &memory, type));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(buffer.Begin()(), memory, cnt, cudaMemcpyHostToDevice));
    }

    void Resize(uint leafs, uint interiorNodes)
    {
        uint nodeCount = interiorNodes + leafs;

        lineartreeNodeIsLeaf.Resize(nodeCount);
        nodeIndexToLeafIndex.Resize(nodeCount);
        lineartreeLeftNode.Resize(nodeCount);
        lineartreeRightNode.Resize(nodeCount);
        lineartreeSplit.Resize(nodeCount);
        lineartreeSplitAxis.Resize(nodeCount);

        lineartreeContentStart.Resize(leafs);
        lineartreeContentCount.Resize(leafs);
    }
};

double g_lastTreeBuildTime = 0;
void updateTree(ICTTree* tree, NodeGPUDataTransformer* gpuData)
{
    chimera::util::HTimer loadTimer;

    {
        print("Building Tree ...\n");
        loadTimer.Start();
        CT_SAFE_CALL(CTUpdate(tree));
        loadTimer.Stop();
        g_lastTreeBuildTime = loadTimer.GetMillis();
        print("Building Tree took '%f' Seconds\n", loadTimer.GetSeconds());
    }    

    CTuint nodeCount;
    CT_SAFE_CALL(CTGetInteriorNodeCount(tree, &nodeCount));

    CTuint leafs;
    CT_SAFE_CALL(CTGetLeafNodeCount(tree, &leafs));

    print("Getting Tree Data ...\n");
    loadTimer.Start();
    
    TreeNodes gpuTree;
    CT_TREE_DEVICE type;
    CT_SAFE_CALL(CTGetTreeDeviceType(tree, &type));

    if(type == eCT_CPU)
    {
        gpuData->Reset();
        gpuData->Resize(leafs, nodeCount);

        gpuData->Fill(tree, eCT_NODE_IS_LEAF, gpuData->lineartreeNodeIsLeaf);
        gpuData->Fill(tree, eCT_LEAF_NODE_PRIM_START_INDEX, gpuData->lineartreeContentStart);
        gpuData->Fill(tree, eCT_LEAF_NODE_PRIM_COUNT, gpuData->lineartreeContentCount);
        gpuData->Fill(tree, eCT_NODE_LEFT_CHILD, gpuData->lineartreeLeftNode);
        gpuData->Fill(tree, eCT_NODE_RIGHT_CHILD, gpuData->lineartreeRightNode);
        gpuData->Fill(tree, eCT_NODE_SPLITS, gpuData->lineartreeSplit);
        gpuData->Fill(tree, eCT_NODE_SPLIT_AXIS, gpuData->lineartreeSplitAxis);
        gpuData->Fill(tree, eCT_NODE_TO_LEAF_INDEX, gpuData->nodeIndexToLeafIndex);

        gpuTree.leafIndex = gpuData->nodeIndexToLeafIndex.Begin()();
        gpuTree.contentCount = gpuData->lineartreeContentCount.Begin()();
        gpuTree.contentStart = gpuData->lineartreeContentStart.Begin()();

        CTuint d;
        CTGetDepth(tree, &d);
        if(d > 0)
        {
            gpuTree._left = gpuData->lineartreeLeftNode.Begin()();
            gpuTree._right = gpuData->lineartreeRightNode.Begin()();
            gpuTree.split = gpuData->lineartreeSplit.Begin()();
            gpuTree.splitAxis = gpuData->lineartreeSplitAxis.Begin()();
        }

        gpuTree.isLeaf = gpuData->lineartreeNodeIsLeaf.Begin()();

        uint cnt;
        const void* memory;
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, &memory, eCT_LEAF_NODE_PRIM_IDS));
        gpuData->lineartreeContent.Resize(cnt / sizeof(uint));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuData->lineartreeContent.Begin()(), memory, cnt, cudaMemcpyHostToDevice));

        gpuTree.content = gpuData->lineartreeContent.Begin()();
    }
    else if(type == eCT_GPU)
    {
        CTuint cnt;
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree.isLeaf, eCT_NODE_IS_LEAF));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree.contentStart, eCT_LEAF_NODE_PRIM_START_INDEX));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree.contentCount, eCT_LEAF_NODE_PRIM_COUNT));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree._left, eCT_NODE_LEFT_CHILD));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree._right, eCT_NODE_RIGHT_CHILD));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree.split, eCT_NODE_SPLITS));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree.splitAxis, eCT_NODE_SPLIT_AXIS));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree.leafIndex, eCT_NODE_TO_LEAF_INDEX));
        CT_SAFE_CALL(CTGetLinearMemory(tree, &cnt, (const void**)&gpuTree.content, eCT_LEAF_NODE_PRIM_IDS));
    }

    loadTimer.Stop();
    print("Traversing Tree took '%f' Seconds\n", loadTimer.GetSeconds());

    RT_BindTree(gpuTree);
}

struct GPUTraceData
{
    nutty::DeviceBuffer<Material> materials;
    nutty::DeviceBuffer<Normal> triNormals;
    nutty::DeviceBuffer<TexCoord> triTc;
    nutty::DeviceBuffer<byte> perVertexMatId;
};

struct GeoHandle
{
    CTuint start;
    CTuint end;
    CTGeometryHandle handle;

    GeoHandle(void)
    {

    }

    GeoHandle(const GeoHandle& gh)
    {
        start = gh.start;
        end = gh.end;
        handle = gh.handle;
    }
};

std::map<std::string, RawTriangles*> g_modelCache;

CTGeometryHandle AddGeometry(GPUTraceData& data, ICTTree* tree, cuTextureAtlas* atlas, const char* objFile, GeoHandle* geoHandle = NULL)
{
    RawTriangles* cpuTris;

    chimera::util::HTimer loadTimer;
    auto it = g_modelCache.find(std::string(objFile));
    if(it != g_modelCache.end())
    {
        cpuTris = it->second;
    }
    else
    {
        RawTriangles* _cpuTris = new RawTriangles();
        std::string modelPath;
        FindFilePath(objFile, modelPath);
        loadTimer.Start();
        if(!ReadObjFileThreaded(modelPath.c_str(), *_cpuTris))
        {
            print("Couldn't load model!\n");
            return -1;
        }
        g_modelCache.insert(std::pair<std::string, RawTriangles*>(std::string(objFile), _cpuTris));
        cpuTris = _cpuTris;
    }

    loadTimer.Stop();
    print("Loading '%s' took '%f' Seconds (Used Cache '%d')\n", objFile, loadTimer.GetSeconds(), (CTuint)(it != g_modelCache.end()));

    size_t tcstart = data.triTc.Size();
    data.triTc.Resize(tcstart + cpuTris->tcoords.size());

    size_t nstart = data.triNormals.Size();
    data.triNormals.Resize(nstart + cpuTris->normals.size());

    size_t matIdstart = data.perVertexMatId.Size();
    
    if(!cpuTris->tcoords.empty())
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(data.triTc.Begin()() + tcstart, &(cpuTris->tcoords[0]), cpuTris->tcoords.size() * sizeof(TexCoord), cudaMemcpyHostToDevice));
    }

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(data.triNormals.Begin()() + nstart, &(cpuTris->normals[0]), cpuTris->normals.size() * sizeof(Normal), cudaMemcpyHostToDevice));

    std::map<std::string, int> texToSlot;
    for(auto& it = cpuTris->materials.begin(); it !=  cpuTris->materials.end(); ++it)
    {
        if(it->second.texFile.size() && texToSlot.find(it->second.texFile) == texToSlot.end())
        {
            std::string texPath;
            FindFilePath(it->second.texFile.c_str(), texPath);
            int slot = atlas->AddTexture(texPath.c_str());
            texToSlot[it->second.texFile] = slot;
        }
    }

    uint matOffset = data.materials.GetPos();

    for(auto& i = cpuTris->materials.begin(); i != cpuTris->materials.end(); ++i)
    {
        RawMaterial cpuMat = i->second;
        Material mat;
        mat.ambientI = cpuMat.ambientI;
        mat.diffuseI = cpuMat.diffuseI;
        mat.specularI = cpuMat.specularI;

        int texIndex = NO_TEXTURE;

        auto it = texToSlot.find(cpuMat.texFile);

        if(it != texToSlot.end())
        {
            texIndex = it->second;
        }

        mat._specExp = cpuMat.specularExp;
        mat._mirror = cpuMat.mirror != 0;
        mat._alpha = cpuMat.alpha;
        mat._texId = texIndex;
        mat._reflectionIndex = cpuMat.ior;
        mat._fresnel_r = cpuMat.fresnel_r;
        mat._fresnel_t = cpuMat.fresnel_t;
        mat._reflectivity = cpuMat.reflectivity;

        data.materials.PushBack(mat);
    }

    CTGeometryHandle handle = 0;

    std::vector<byte> matsIds;

    for(auto& i = cpuTris->intervals.begin(); i != cpuTris->intervals.end(); ++i)
    {
        matsIds.reserve(matsIds.size() + i->end - i->start);
        byte matIndex = matOffset + cpuTris->GetMaterialIndex(i->material);
        ICTGeometry* geo;
        CTCreateGeometry(&geo);
        for(uint a = i->start; a < i->end; ++a)
        { 
            CTTriangle tri;
            for(int k = 0; k < 3; ++k)
            {
                matsIds.push_back(matIndex);
                CTreal3 pos;
                Position p = cpuTris->positions[3 * a + k];
                pos.x = p.x;
                pos.y = p.y;
                pos.z = p.z;
                tri.SetValue(k, pos);
            }
            CTAddPrimitive(geo, &tri);
        }
        //CTAddGeometry(tree, geo, &handle);
        CTreal3* ptr = &cpuTris->positions[0];
        CT_SAFE_CALL(CTAddGeometryFromLinearMemory(tree, ptr + 3*i->start, 3*(i->end - i->start), &handle));
        if(geoHandle)
        {
            geoHandle->handle = handle;
            geoHandle->start = 3*i->start;
            geoHandle->end = 3*i->end;

        }
    }

    uint start = data.perVertexMatId.Size();
    data.perVertexMatId.Resize(data.perVertexMatId.Size() + matsIds.size());
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(data.perVertexMatId.Begin()() + start, &matsIds[0], matsIds.size() * sizeof(byte), cudaMemcpyHostToDevice));

    return handle;
}

void createTestTree(CTuint type)
{
    ICTTree* tree;
    CT_SAFE_CALL(CTCreateSAHKDTree(&tree, type));

    GPUTraceData triGPUData;
    GeoHandle hhandle;
    AddGeometry(triGPUData, tree, atlas, "empty_room.obj", &hhandle);

    std::vector<GeoHandle> cubeHandles;

    chimera::util::Mat4 model;
    CTuint addSum = 12*3;
    uint c = 16;
    for(int i = 0; i < c; ++i)
    {
        int s = (int)sqrt(c);
        CTGeometryHandle handle = AddGeometry(triGPUData, tree, atlas, "cube.obj", &hhandle);
        CTuint sumcopy = addSum;
        hhandle.start += addSum;
        hhandle.end += addSum;
        cubeHandles.push_back(hhandle);
        addSum += (hhandle.end - hhandle.start);
        //model.SetTranslation(0,4,0);
        model.SetTranslation(-6 + 3*(i/s), 4, -6 + 3*(i%s));
        CT_SAFE_CALL(CTTransformGeometryHandle(tree, hhandle.handle, (CTreal4*)model.m_m.m));
    }

    //CTGeometryHandle handle = AddGeometry(triGPUData, tree, atlas, "mikepan_bmw3v3.obj");

    CT_SAFE_CALL(CTUpdate(tree));

    CT_SAFE_CALL(CTReleaseTree(tree));
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR str, int nCmdShow)
{

// #ifdef _DEBUG
// #define _CRTDBG_MAP_ALLOC
//     _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_CHECK_CRT_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_EVERY_16_DF);
// #endif

    width = (1024 * 3) / 2;
    height = (512 * 3) / 2;
    
    HWND hwnd = CreateScreen(hInstance, WndProc, MY_CLASS, L"", width, height);

    RECT rect;
    GetClientRect(hwnd, &rect);
    
    HGLRC context; 
    if(!(context = CreateGLContextAndMakeCurrent(hwnd)))
    {
        return -1;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);  
    Resize(width, height);

    HDC hDC = GetDC(hwnd);

    glClearColor(0,0,0,0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SwapBuffers(hDC);

    g_cam.Move(0, 5, -5);

    assert(FontInit());

    glProgram* p = glProgram::CreateProgramFromFile("glsl/vs.glsl", "glsl/fs.glsl");
    p->Bind();

    int twidth = width / 2;
    int theight = height;
    
    int size = twidth * theight * 4;

    glTextureBuffer* tb = new glTextureBuffer();

    tb->Resize(size * sizeof(float));

    tb->BindToTextureSlot(TEXTURE_SLOT_RT_COLOR_BUFFER);

    checkGLError();

    nutty::Init();
    
    cudaGraphicsResource_t res = NULL;    
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsGLRegisterBuffer(&res, tb->BufferId(), 0));

    glUniform1i(glGetUniformLocation(p->Id(), "sampler"), TEXTURE_SLOT_RT_COLOR_BUFFER);

    glUniform1i(glGetUniformLocation(p->Id(), "width"), twidth);

    RT_Init(twidth, theight);

    atlas = new cuTextureAtlas();
    atlas->Init();

    CT_SAFE_CALL(CTInit(0));
    
//   createTestTree(CT_CREATE_TREE_CPU);
//   print("\n\n\n");
//   createTestTree(CT_CREATE_TREE_GPU);

    //return 0;
    uint treeType = CT_CREATE_TREE_GPU;
    ICTTree* tree;
    CT_SAFE_CALL(CTCreateSAHKDTree(&tree, treeType));

    GPUTraceData* triGPUData = new GPUTraceData();
    GeoHandle hhandle;
    chimera::util::Mat4 model;
    AddGeometry(*triGPUData, tree, atlas, "empty_room_big.obj", &hhandle);
    //model.SetScale(10,10,10);
    //CT_SAFE_CALL(CTTransformGeometryHandle(tree, hhandle.handle, (CTreal4*)model.m_m.m));

    std::vector<GeoHandle> cubeHandles;
    chimera::util::cmRNG rng;
    CTuint addSum = 12*3;
    int line = 1;
    srand(0);

    for(int i = 0; i < line; ++i)
    {
        for(int j = 0; j < line; ++j)
        {
            for(int k = 0; k < line; ++k)
            {
                CTGeometryHandle handle = AddGeometry(*triGPUData, tree, atlas, "cube.obj", &hhandle); //"ice_cube_small.obj"
                CTuint sumcopy = addSum;
                hhandle.start += addSum;
                hhandle.end += addSum;
                cubeHandles.push_back(hhandle);
                addSum += (hhandle.end - hhandle.start);
                //model.SetTranslation(rand() % 10,4,rand() % 10);
                float scale = 4;
                float ox = rng.NextCubeFloat(2.1f);
                float oy = rng.NextFloat(1.1f);
                float oz = rng.NextCubeFloat(2.1f);
                model.SetTranslation(-scale*(line/2) + scale * j, 3 + scale * i, -scale*(line/2) + scale * k);
                model.Translate(ox, oy, oz);
                CT_SAFE_CALL(CTTransformGeometryHandle(tree, hhandle.handle, (CTreal4*)model.m_m.m));
            }
        }
    }

    //CTGeometryHandle handle = AddGeometry(*triGPUData, tree, atlas, "Spiral_Caged.obj");
    //CTGeometryHandle handle = AddGeometry(*triGPUData, tree, atlas, "mikepan_bmw3v3.obj");

    nutty::DeviceBuffer<Normal> normalsSave(triGPUData->triNormals.Size());
    nutty::Copy(normalsSave.Begin(), triGPUData->triNormals.Begin(), triGPUData->triNormals.End());

    uint vertexCount;
    CT_SAFE_CALL(CTGetPrimitiveCount(tree, &vertexCount));
    vertexCount *= 3;

    NodeGPUDataTransformer* nodeGPUData = new NodeGPUDataTransformer();

    Triangles tris;
    tris.materials = triGPUData->materials.Begin()();
    tris.normals = triGPUData->triNormals.Begin()();
    tris.texCoords = triGPUData->triTc.Begin()();
    tris.matId = triGPUData->perVertexMatId.Begin()();

    uint count;
    const cuTextureObj* textures = atlas->GetTextures(&count);
    RT_BindTextures(textures, count);

    debugLayer = new glDebugLayer();
    debugLayer->GetProgram()->Bind();

    GLuint loc = glGetUniformLocation(debugLayer->GetProgram()->Id(), "perspective");
    glUniformMatrix4fv(loc, 1, false, (float*)g_cam.GetProjection());
    std::stringstream ss;

    RT_SetViewPort(twidth, theight);

    chimera::util::Mat4 matrix;
    chimera::util::Mat4 cameraMatrix;
    float time = 0;

    //geoTransMatrix.SetTranslation(0,1,0);
    //while(1) CTUpdate(tree);

    updateTree(tree, nodeGPUData);

    if(treeType == CT_CREATE_TREE_CPU)
    {
        debugLayer->ResetGeometry();
        CT_SAFE_CALL(CTTreeDrawDebug(tree, debugLayer)); //collectdata
    }

    const void* memory = NULL;
    CTuint cnt;
    CT_SAFE_CALL(CTGetRawLinearMemory(tree, &cnt, &memory));

    if(treeType == CT_CREATE_TREE_CPU)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&tris.positions, vertexCount * sizeof(Position)));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(tris.positions, memory, cnt, cudaMemcpyHostToDevice));
    }
    else
    {
        tris.positions = (Position*)memory;
    }

    RT_BindGeometry(tris);

    MSG msg;
    chimera::util::HTimer traverseTimer;
    chimera::util::HTimer traceTimer;
    chimera::util::HTimer timer;
    timer.VReset();
    chimera::util::Vec3 v;

    while(1)
    {
        if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if(msg.message == WM_QUIT)
            {
                break;
            }
        } 
        else 
        { 
            float time = timer.VGetTime() * 2 * 1e-4f;
            float dt = 1e-3f * timer.VGetLastMillis();
            static bool a = !true;
            if(a)
              {
                  int index = 0;
//                   for(int i = 0; i < line; ++i)
//                   {
//                       for(int j = 0; j < line; ++j)
//                       {
//                           for(int k = 0; k < line; ++k)
//                           {
//                               GeoHandle hhandle = cubeHandles[index];
//                               float dir = -1 + 2 * (index % 2);
//                               model.SetRotateX(dir * (time + index/(float)cubeHandles.size()));
//                               model.RotateY(dir * (time + 0.5f*index/(float)cubeHandles.size()));
//                               model.RotateZ(dir * (time + 2*index/(float)(float)cubeHandles.size()));
//                               float scale = 4;
//                               model.SetTranslation(-scale*(line/2) + scale * j, 3 + scale * i, -scale*(line/2) + scale * k);
//        
//                               CT_SAFE_CALL(CTTransformGeometryHandle(tree, hhandle.handle, (CTreal4*)model.m_m.m));
// 
//                               RT_TransformNormals(normalsSave.Begin()(), tris.normals, (CTreal4*)model.m_m.m, hhandle.start, (hhandle.end - hhandle.start));
//                               index++;
//                           }
//                       }
//                 }
                cudaDeviceSynchronize();
                traverseTimer.Start();
                updateTree(tree, nodeGPUData);
                traverseTimer.Stop();
                if(treeType == CT_CREATE_TREE_CPU)
                {
                    const void* memory = NULL; 
                    CTuint cnt;
                    CT_SAFE_CALL(CTGetRawLinearMemory(tree, &cnt, &memory));
                    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(tris.positions, memory, cnt, cudaMemcpyHostToDevice));
                }
                //cudaDeviceSynchronize();
            }
  
            computeMovement(dt);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glViewport(width - twidth, 0, width - twidth, height);

            debugLayer->BeginDraw();
            GLuint loc = glGetUniformLocation(debugLayer->GetProgram()->Id(), "view");
            glUniformMatrix4fv(loc, 1, false, (float*)g_cam.GetIView());
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
//             if(a)
//             {
//                 debugLayer->ResetGeometry();
//                 CT_SAFE_CALL(CTTreeDrawDebug(tree, debugLayer));
//             }
            debugLayer->DrawGLGeo();

            glDisable(GL_BLEND);
            debugLayer->EndDraw();

            int glError = glGetError();
            if(glError != GL_NO_ERROR)
            {
                print("%d\n", glError);
            }

            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsMapResources(1, &res));

            float4* mappedPtr;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsResourceGetMappedPointer((void**)&mappedPtr, NULL, res));
            
            const ICTAABB* aabb;
            CT_SAFE_CALL(CTGetAxisAlignedBB(tree, &aabb));
            BBox bbox;
            bbox.addPoint(aabb->GetMin());
            bbox.addPoint(aabb->GetMax());

            traceTimer.Start();
            RT_Trace(mappedPtr, g_cam.GetView(), g_cam.GetEye(), bbox);

            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnmapResources(1, &res));

            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSynchronize());

            traceTimer.Stop();
            
            //font
            uint inc;
            uint lnc;
            CTGetInteriorNodeCount(tree, &inc);
            CTGetLeafNodeCount(tree, &lnc);
            ss.str("");
            ss << "Nodes=" << (inc + lnc) << "\n";
            ss << "Leafes=" << lnc << "\n";

            double fps = (int)((1000 / (double)traceTimer.GetMillis()));

            double fps_all = (int)((1000 / (double)(traceTimer.GetMillis() + traverseTimer.GetMillis())));
            ss << twidth * theight << " Pixel\n";
            ss << RT_GetLastRayCount() << " Rays\n";
            ss << vertexCount/3 << " Primitives\n";
            //ss << (int)(1000.0 / (traverseTimer.GetMillis() == 0 ? 1 : traverseTimer.GetMillis())) << " FPS (Build)\n";
            ss << fps << " FPS (Tracing)\n";
            ss << (int)(1000 / g_lastTreeBuildTime) << " FPS (" << ((treeType == CT_CREATE_TREE_CPU) ? "CPU" : "GPU") << " Build)\n";
            ss << fps_all << " FPS (Overall)";

            //             ss << "\n\nMirror=" << g_matToPrint->isMirror() << " (1)\n";
            //             ss << "Alpha=" << g_matToPrint->alpha() << " (2,3)\n";
            //             ss << "Reflectance=" << g_matToPrint->reflectivity() << " (4,5)\n";
            //             ss << "Fresnel_R=" << g_matToPrint->fresnel_r() << " (6,7)\n";
            //             ss << "Fresnel_T=" << g_matToPrint->fresnel_t() << " (8,9)\n";
            //             ss << "IOR=" << g_matToPrint->reflectionIndex() << " (o,p)\n";

            FontBeginDraw();
            FontDrawText(ss.str().c_str(), 0.0025, 0);
            std::string info;
            RT_GetRayInfo(info);
            FontDrawText(info.c_str(), 0.0025, 0.35f);
            FontEndDraw();

            drawUI();
            //font end

            p->Bind();
            glViewport(0, 0, twidth, theight);
            glBindVertexArray(0);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glDrawArrays(GL_TRIANGLES, 0, 4);

            SwapBuffers(hDC);

            float3 vp;
            if(g_animateLight)
            {
                v.Set(0, 20, -20);

                //matrix.RotateX(0.25*dt);
                matrix.RotateY(dt);

                v = chimera::util::Mat4::Transform(matrix, v);

                vp.x = v.GetX();
                vp.y = v.GetY();
                vp.z = v.GetZ();

                RT_SetLightPos(vp);
            }

            if(g_animateCamera)
            {
                float3 lookAt;
                lookAt.x = 0;
                lookAt.y = 3;
                lookAt.z = 0;

                v.Set(0, 4, -10);
                cameraMatrix.RotateY(-0.25*dt);

                v = chimera::util::Mat4::Transform(cameraMatrix, v);
                vp.x = v.GetX();
                vp.y = v.GetY();
                vp.z = v.GetZ();
                g_cam.LookAt(lookAt, vp);
            }
            timer.VTick();
        }
    }

    for(auto& it = g_modelCache.begin(); it != g_modelCache.end(); ++it)
    {
        delete it->second;
    }

    delete debugLayer;
    delete triGPUData;
    delete nodeGPUData;
    
    RT_Destroy();
    
    delete atlas;
    cudaFree(tris.materials);
    
    if(treeType == CT_CREATE_TREE_CPU)
    {
        cudaFree(tris.positions);
    }

    cudaFree(tris.normals);
    cudaFree(tris.texCoords);
    cudaFree(tris.matId);

    cudaGraphicsUnregisterResource(res);

    delete p;
    delete tb;

    CT_SAFE_CALL(CTRelease());
    nutty::Release();

    FontDestroy();

    ReleaseGLContext(context);

    ReleaseScreen(hInstance, hwnd, MY_CLASS);

    return 0;
}