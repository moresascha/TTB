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

#include <Nutty.h>

#include <chimera/api/ChimeraAPI.h>
#include <chimera/Vec4.h>
#include <chimera/Timer.h>
#include <chimera/Vec3.h>
#include <chimera/Mat4.h>

#include "cpuKDTree.h"

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

glDebugLayer* debugLayer = NULL;

std::vector<Material> cpuMats;
Material* gpuMats = NULL;

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
bool g_animateLight = true;

bool computeMovement(void)
{
    if(!g_isKeyDown)
    {
        return false;
    }

    float delta = 0.25f;
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

void computeMaterialChange(void)
{
    //97 ->
    Material& mat = cpuMats[0];

    if(g_key == KEY_G)
    {
        g_animateCamera ^= 1;
    }
    else if(g_key == KEY_H)
    {
        g_animateLight ^= 1;
    }
    else if(g_key == 97)
    {
        mat._mirror ^= 1;
    }

    else if(g_key == KEY_J)
    {
        RT_IncDepth();
    }
    else if(g_key == KEY_K)
    {
        RT_DecDepth();
    }

    //Alpha
    else if(g_key == 98)
    {
        mat._alpha -= 0.01f;
    }
    else if(g_key == 99)
    {
        mat._alpha += 0.01f;
    }

    //Reflectance
    else if(g_key == 100)
    {
        mat._reflectivity -= 0.01f;
    }
    else if(g_key == 101)
    {
        mat._reflectivity += 0.01f;
    }

    //Fresnel_R
    else if(g_key == 102)
    {
        mat._fresnel_r -= 0.01f;
    }
    else if(g_key == 103)
    {
        mat._fresnel_r += 0.01f;
    }

    //Fresnel_T
    else if(g_key == 104)
    {
        mat._fresnel_t -= 0.01f;
    }
    else if(g_key == 105)
    {
        mat._fresnel_t += 0.01f;
    }

    //IOR
    else if(g_key == KEY_P)
    {
        mat._reflectionIndex -= 0.01f;
    }
    else if(g_key == KEY_O)
    {
        mat._reflectionIndex += 0.01f;
    }
    
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuMats, (void*)&cpuMats[0], sizeof(Material), cudaMemcpyHostToDevice));
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

void traverse(cpuTreeNode* node, uint d)
{
    for(int i = 0; i < d; ++i) print(" ");

    if(node->isLeaf)
    {
        print("Leaf\n");
        return;
    }

    print("Node\n");

    traverse(node->left, d+1);
    traverse(node->right, d+1);
}

void checkGLError(void)
{
    int glError = glGetError();
    if(glError != GL_NO_ERROR)
    {
        print("%d\n", glError);
    }
}

uint currentLeafIndex = 0;
void collectDataDFO(cpuTreeNode* node, 
                 std::vector<uint>& nodeIndexToLeafIndex,
                 std::vector<uint>& lineartreeContent,
                 std::vector<uint>& lineartreeContentCount,
                 std::vector<uint>& lineartreeContentStart,
                 std::vector<uint>& lineartreeLeftNode, 
                 std::vector<uint>& lineartreeRightNode, 
                 std::vector<float>& lineartreeSplit,
                 std::vector<byte>& lineartreeSplitAxis, 
                 std::vector<byte>& lineartreeNodeIsLeaf)
{
    lineartreeNodeIsLeaf.push_back(node->isLeaf);
    nodeIndexToLeafIndex.push_back(currentLeafIndex);

    lineartreeLeftNode.push_back(node->leftAdd);
    lineartreeRightNode.push_back(node->rightAdd);
    lineartreeSplit.push_back(node->split);
    lineartreeSplitAxis.push_back((byte)node->splitAxis);

    if(node->isLeaf)
    {
        lineartreeContentStart.push_back(lineartreeContent.size());
        lineartreeContentCount.push_back(node->geometry.size());

        currentLeafIndex++;
        for(int i = 0; i < node->geometry.size(); ++i)
        {
            lineartreeContent.push_back(node->geometry[i]);
        }
        return;
    }

    collectDataDFO(node->left, 
        nodeIndexToLeafIndex, 
        lineartreeContent, 
        lineartreeContentCount, 
        lineartreeContentStart, 
        lineartreeLeftNode, 
        lineartreeRightNode, 
        lineartreeSplit, 
        lineartreeSplitAxis, 
        lineartreeNodeIsLeaf);

    collectDataDFO(node->right, 
        nodeIndexToLeafIndex, 
        lineartreeContent, 
        lineartreeContentCount, 
        lineartreeContentStart, 
        lineartreeLeftNode, 
        lineartreeRightNode, 
        lineartreeSplit, 
        lineartreeSplitAxis, 
        lineartreeNodeIsLeaf);
}

void collectDataBFO(cpuTreeNode* node, 
                    std::vector<uint>& nodeIndexToLeafIndex,
                    std::vector<uint>& lineartreeContent,
                    std::vector<uint>& lineartreeContentCount,
                    std::vector<uint>& lineartreeContentStart,
                    std::vector<uint>& lineartreeLeftNode, 
                    std::vector<uint>& lineartreeRightNode, 
                    std::vector<float>& lineartreeSplit,
                    std::vector<byte>& lineartreeSplitAxis, 
                    std::vector<byte>& lineartreeNodeIsLeaf)
{
    std::queue<cpuTreeNode*> queue;

    queue.push(node);

    uint address = 1;
    uint leafIndex = 0;
    uint left = 0;
    while(!queue.empty())
    {
        cpuTreeNode* node = queue.front();
        queue.pop();

        if(!node->visited)
        {
            lineartreeNodeIsLeaf.push_back(node->isLeaf);
            nodeIndexToLeafIndex.push_back(currentLeafIndex);

            lineartreeLeftNode.push_back(node->isLeaf ? -1 : address+0);
            lineartreeRightNode.push_back(node->isLeaf ? -1 : address+1);
            lineartreeSplit.push_back(node->split);
            lineartreeSplitAxis.push_back((byte)node->splitAxis);

            node->visited = true;

            if(!node->isLeaf)
            {
                queue.push(node->left);
                queue.push(node->right);
                address += 2;
            }
            else
            {
                leafIndex++;
                lineartreeContentStart.push_back(lineartreeContent.size());
                lineartreeContentCount.push_back(node->geometry.size());

                currentLeafIndex++;
                for(int i = 0; i < node->geometry.size(); ++i)
                {
                    lineartreeContent.push_back(node->geometry[i]);
                }
            }
        }
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR str, int nCmdShow)
{
    width = (1024 * 3) / 2;
    height = (512 * 3) / 2;

    HWND hwnd = CreateScreen(hInstance, WndProc, MY_CLASS, L"", width, height);

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

    MSG msg;

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

    cudaGraphicsResource_t res;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsGLRegisterBuffer(&res, tb->BufferId(), 0));

    glUniform1i(glGetUniformLocation(p->Id(), "sampler"), TEXTURE_SLOT_RT_COLOR_BUFFER);

    glUniform1i(glGetUniformLocation(p->Id(), "width"), twidth);

    RawTriangles cpuTris;
    std::string modelPath;
    FindFilePath("dragon.obj", modelPath);
    chimera::util::HTimer loadTimer;
    loadTimer.Start();
    if(!ReadObjFileThreaded(modelPath.c_str(), cpuTris))
    {
        print("Couldn't load model!\n");
        return -1;
    }
    loadTimer.Stop();
    print("Loading took '%f' Seconds\n", loadTimer.GetSeconds());
    int vertexCount = (int)cpuTris.positions.size();

    BBox bbox;
    bbox.init();

    for(int i = 0; i < vertexCount / 3; ++i)
    {
        bbox.addPoint(cpuTris.positions[3 * i + 0]);
        bbox.addPoint(cpuTris.positions[3 * i + 1]);
        bbox.addPoint(cpuTris.positions[3 * i + 2]);
    }

    cuTextureAtlas* atlas = new cuTextureAtlas();
    atlas->Init();

    std::map<std::string, int> texToSlot;
    for(auto& it = cpuTris.materials.begin(); it !=  cpuTris.materials.end(); ++it)
    {
        if(it->second.texFile.size() && texToSlot.find(it->second.texFile) == texToSlot.end())
        {
            std::string texPath;
            FindFilePath(it->second.texFile.c_str(), texPath);
            int slot = atlas->AddTexture(texPath.c_str());
            texToSlot[it->second.texFile] = slot;
        }
    }

    Triangles tris;
    tris.colors = NULL;
    tris.materials = NULL;
    tris.normals = NULL;
    tris.positions = NULL;
    tris.texCoords = NULL;
    byte* gpuTriMatIds = NULL;
    Position* gpuPos = NULL;
    Normal* gpuNorm = NULL;
    TexCoord* gpuTc = NULL;

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuMats, cpuTris.materials.size() * sizeof(Material)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuPos, vertexCount * sizeof(Position)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuNorm, vertexCount * sizeof(Normal)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTc, vertexCount * sizeof(TexCoord)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTriMatIds, vertexCount * sizeof(byte)));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuPos, (void*)&cpuTris.positions[0], vertexCount * sizeof(Position), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuNorm, (void*)&cpuTris.normals[0], vertexCount * sizeof(Normal), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTc, (void*)&cpuTris.tcoords[0], vertexCount * sizeof(TexCoord), cudaMemcpyHostToDevice));
    
    for(auto& i = cpuTris.materials.begin(); i != cpuTris.materials.end(); ++i)
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

        cpuMats.push_back(mat);
    }

    g_matToPrint = &cpuMats[0];

    std::vector<byte> matIds;
    for(auto& i = cpuTris.intervals.begin(); i != cpuTris.intervals.end(); ++i)
    {
        byte matIndex = cpuTris.GetMaterialIndex(i->material);
        for(int a = i->start; a < i->end; ++a)
        {
            for(int k = 0; k < 3; ++k)
            {
                matIds.push_back(matIndex);
            }
        }
    }

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuMats, (void*)&cpuMats[0], cpuTris.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTriMatIds, (void*)&matIds[0], vertexCount * sizeof(byte), cudaMemcpyHostToDevice));

    tris.materials = gpuMats;
    tris.normals = gpuNorm;
    tris.positions = gpuPos;
    tris.texCoords = gpuTc;
    tris.matId = gpuTriMatIds;

    RT_Init(twidth, theight);

    CT_SAFE_CALL(CTInit(CT_ENABLE_CUDA_ACCEL | CT_TREE_ENABLE_DEBUG_LAYER));

    ICTTree* tree;
    CT_SAFE_CALL(CTCreateSAHKDTree(&tree, CT_CREATE_TREE_CPU));

    for(int i = 0; i < vertexCount / 3; ++i)
    {
        ICTGeometry* geo;
        CT_SAFE_CALL(CTCreateGeometry(&geo));

        for(int j = 0; j < 3; ++j)
        {
            ICTVertex* v;
            CT_SAFE_CALL(CTCreateVertex(&v));
            ctfloat3 pos;
            Position p = cpuTris.positions[3 * i + j];
            pos.x = p.x;
            pos.y = p.y;
            pos.z = p.z;
            v->SetPosition(pos);
            geo->AddVertex(v);
        }

        CT_SAFE_CALL(tree->AddGeometry(geo));
    }

    /*ICTMemoryView* inter;
    CT_SAFE_CALL(tree->QueryInterface(__uuidof(ICTMemoryView), (void**)&inter));

    ct_printf("%x\n", inter->GetMemory());*/

    ICTTreeNode* node = tree->GetNodesEntryPtr();
    
    loadTimer.Start();
    CT_SAFE_CALL(tree->Update());
    loadTimer.Stop();
    print("Building Tree took '%f' Seconds\n", loadTimer.GetSeconds());

    std::vector<uint> nodeIndexToLeafIndex;
    std::vector<uint> lineartreeContent;
    std::vector<uint> lineartreeContentCount;
    std::vector<uint> lineartreeContentStart;
    std::vector<uint> lineartreeLeftNode;
    std::vector<uint> lineartreeRightNode;

    std::vector<float> lineartreeSplit;
    std::vector<byte> lineartreeSplitAxis;
    std::vector<byte> lineartreeNodeIsLeaf;

    loadTimer.Start();
#if 0
    collectDataDFO((cpuTreeNode*)node,
        nodeIndexToLeafIndex,
        lineartreeContent,
        lineartreeContentCount,
        lineartreeContentStart,
        lineartreeLeftNode,
        lineartreeRightNode,
        lineartreeSplit,
        lineartreeSplitAxis,
        lineartreeNodeIsLeaf);
#else
    collectDataBFO((cpuTreeNode*)node,
        nodeIndexToLeafIndex,
        lineartreeContent,
        lineartreeContentCount,
        lineartreeContentStart,
        lineartreeLeftNode,
        lineartreeRightNode,
        lineartreeSplit,
        lineartreeSplitAxis,
        lineartreeNodeIsLeaf);
#endif
    loadTimer.Stop();
    print("Traversing Tree took '%f' Seconds\n", loadTimer.GetSeconds());

#if 0
    for(int i = 0; i < nodeIndexToLeafIndex.size(); ++i)
    {
        print("%d -> %d\n", i, nodeIndexToLeafIndex[i]);
    }

    for(int i = 0; i < nodeIndexToLeafIndex.size(); ++i)
    {
        print("left=%d right=%d\n", lineartreeLeftNode[i], lineartreeRightNode[i]);
    }

    for(int i = 0; i < lineartreeContentCount.size(); ++i)
    {
        print("%d -> %d\n", lineartreeContentStart[i], lineartreeContentCount[i]);
    }

    for(int i = 0; i < lineartreeNodeIsLeaf.size(); ++i)
    {
        print("isleaf=%d\n", (int)lineartreeNodeIsLeaf[i]);
    }

    for(int i = 0; i < lineartreeSplit.size(); ++i)
    {
        print("%f %d\n", lineartreeSplit[i], lineartreeSplitAxis[i]);
    }
#endif

    TreeNodes gpuTree;

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree.leafIndex, nodeIndexToLeafIndex.size() * sizeof(uint)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree.contentCount, lineartreeContentCount.size() * sizeof(uint)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree.contentStart, lineartreeContentStart.size() * sizeof(uint)));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree.content, lineartreeContent.size() * sizeof(uint)));

    if(tree->GetDepth() > 0)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree._left, lineartreeLeftNode.size() * sizeof(uint)));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree._right, lineartreeRightNode.size() * sizeof(uint)));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree.split, lineartreeSplit.size() * sizeof(float)));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree.splitAxis, lineartreeSplitAxis.size() * sizeof(SplitAxis)));
    }

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&gpuTree.isLeaf, lineartreeNodeIsLeaf.size() * sizeof(byte)));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree.leafIndex, (void*)&nodeIndexToLeafIndex[0], nodeIndexToLeafIndex.size() * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree.contentCount, (void*)&lineartreeContentCount[0], lineartreeContentCount.size() * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree.contentStart, (void*)&lineartreeContentStart[0], lineartreeContentStart.size() * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree.content, (void*)&lineartreeContent[0], lineartreeContent.size() * sizeof(uint), cudaMemcpyHostToDevice));

    if(tree->GetDepth() > 0)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree._left, (void*)&lineartreeLeftNode[0], lineartreeLeftNode.size() * sizeof(uint), cudaMemcpyHostToDevice));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree._right, (void*)&lineartreeRightNode[0], lineartreeRightNode.size() * sizeof(uint), cudaMemcpyHostToDevice));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree.split, (void*)&lineartreeSplit[0], lineartreeSplit.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree.splitAxis, (void*)&lineartreeSplitAxis[0], lineartreeSplitAxis.size() * sizeof(byte), cudaMemcpyHostToDevice));
    }

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuTree.isLeaf, (void*)&lineartreeNodeIsLeaf[0], lineartreeNodeIsLeaf.size() * sizeof(byte), cudaMemcpyHostToDevice));

    RT_BindTree(gpuTree);

    RT_BindGeometry(tris);

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

    //tree->DebugDraw(debugLayer); //collectdata

    uint leafes = nodeIndexToLeafIndex[nodeIndexToLeafIndex.size() - 1];
    uint nodes = lineartreeNodeIsLeaf.size();
    DWORD end = 1;
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
            DWORD start = timeGetTime();
            computeMovement();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glViewport(width - twidth, 0, width - twidth, height);

//             debugLayer->BeginDraw();
//             GLuint loc = glGetUniformLocation(debugLayer->GetProgram()->Id(), "view");
//             glUniformMatrix4fv(loc, 1, false, (float*)g_cam.GetIView());
//             glEnable(GL_BLEND);
//             glBlendFunc(GL_ONE, GL_ONE);
//             debugLayer->DrawGLGeo();
//             glDisable(GL_BLEND);

            ss.str("");
            ss << "MaxDepth=" << tree->GetDepth() << "\n";
            ss << "Nodes=" << nodes << "\n";
            ss << "Leafes=" << leafes << "\n";
            FontBeginDraw();
            FontDrawText(ss.str().c_str(), 0, 0);
            FontEndDraw();

            int glError = glGetError();
            if(glError != GL_NO_ERROR)
            {
                print("%d\n", glError);
            }
            DWORD tr_start = timeGetTime();
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsMapResources(1, &res));

            float4* mappedPtr;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsResourceGetMappedPointer((void**)&mappedPtr, NULL, res));

            RT_Trace(mappedPtr, g_cam.GetView(), g_cam.GetEye(), bbox);

            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnmapResources(1, &res));

            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSynchronize());
   
            p->Bind();
            glViewport(0, 0, twidth, theight);
            glBindVertexArray(0);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glDrawArrays(GL_TRIANGLES, 0, 4);

            DWORD tr_end = max(timeGetTime() - tr_start, 1);
            ss.str("");
            double fps = (int)((1000 / (double)tr_end));
            double fps_all = (int)((1000 / (double)end));
            ss << twidth * theight << " Pixel\n";
            ss << RT_GetLastRayCount() << " Rays\n";
            ss << vertexCount/3 << " Primitives\n";
            ss << fps << " FPS (Tracing)\n";
            ss << fps_all << " FPS";

            ss << "\n\nMirror=" << g_matToPrint->isMirror() << " (1)\n";
            ss << "Alpha=" << g_matToPrint->alpha() << " (2,3)\n";
            ss << "Reflectance=" << g_matToPrint->reflectivity() << " (4,5)\n";
            ss << "Fresnel_R=" << g_matToPrint->fresnel_r() << " (6,7)\n";
            ss << "Fresnel_T=" << g_matToPrint->fresnel_t() << " (8,9)\n";
            ss << "IOR=" << g_matToPrint->reflectionIndex() << " (o,p)\n";

            FontBeginDraw();
            FontDrawText(ss.str().c_str(), 0.0025, 0);

            std::string info;
            RT_GetRayInfo(info);

            FontDrawText(info.c_str(), 0.0025, 0.67f);

            FontEndDraw();

            SwapBuffers(hDC);

            float dt = 0.01f;
            time += dt;
            chimera::util::Vec3 v;
            float3 vp;
            if(g_animateLight)
            {
                v.Set(0,15,-5);

                matrix.RotateX(dt);
                matrix.RotateY(2*dt);

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

                v.Set(0, 4, -9);
                cameraMatrix.RotateY(-dt);

                v = chimera::util::Mat4::Transform(cameraMatrix, v);
                vp.x = v.GetX();
                vp.y = v.GetY();
                vp.z = v.GetZ();
                g_cam.LookAt(lookAt, vp);
            }
            end = max(timeGetTime() - start, 1);
        }
    }

    delete debugLayer;

    cudaFree(gpuTree.contentCount);
    cudaFree(gpuTree.contentStart);
    cudaFree(gpuTree.content);
    cudaFree(gpuTree._left);
    cudaFree(gpuTree._right);
    cudaFree(gpuTree.leafIndex);

    cudaFree(gpuTree.isLeaf);
    cudaFree(gpuTree.split);
    cudaFree(gpuTree.splitAxis);

    RT_Destroy();
    
    delete atlas;
    cudaFree(gpuMats);
    cudaFree(gpuPos);
    cudaFree(gpuNorm);
    cudaFree(gpuTc);
    cudaFree(gpuTriMatIds);

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