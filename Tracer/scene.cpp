
#include "scene.h"
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
#include "tree.h"

#include "glTreeDebug.h"
#include "input.h"

#include "cuda/tracer_api.cuh"

struct GPUTraceData
{
    nutty::DeviceBuffer<Material> materials;
    nutty::DeviceBuffer<Normal> triNormals;
    nutty::DeviceBuffer<TexCoord> triTc;
    nutty::DeviceBuffer<byte> perVertexMatId;
};

struct NodeGPUDataTransformer
{
    nutty::DeviceBuffer<uint> nodeIndexToLeafIndex;
    nutty::DeviceBuffer<uint> lineartreeContent;
    nutty::DeviceBuffer<uint> lineartreeContentCount;
    nutty::DeviceBuffer<uint> lineartreeContentStart;
    nutty::DeviceBuffer<uint> lineartreeLeftNode;
    nutty::DeviceBuffer<uint> lineartreeRightNode;

    nutty::DeviceBuffer<float> lineartreeSplit;
    nutty::DeviceBuffer<CTaxis_t> lineartreeSplitAxis;
    nutty::DeviceBuffer<CTnodeIsLeaf_t> lineartreeNodeIsLeaf;

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
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyAsync(buffer.Begin()(), memory, cnt, cudaMemcpyHostToDevice, tree->GetStream()));
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

class BaseScene : public IScene
{
protected:
    chimera::util::HTimer traceTimer;
    chimera::util::HTimer timer;
    std::map<std::string, RawTriangles*> m_modelCache;
    Camera m_camera;
    cuTextureAtlas* m_atlas;
    glDebugLayer* m_debugLayer;
    int m_w;
    int m_h;
    int m_tw;
    int m_th;
    ICTTree* m_tree;
    NodeGPUDataTransformer* m_nodeGPUData;
    GPUTraceData* m_triGPUData;
    int keyDown_key;
    glProgram* m_p;
    glTextureBuffer* m_tb;
    Triangles m_tris;
    cudaGraphicsResource_t m_res;
    double m_lastTreeBuildTime;
    bool m_updateTreeEachFrame;
    bool m_animateGeometry;
    float m_envMapScale;
    bool m_drawDebugTree;
    bool m_drawFont;
    Material m_matToPrint;
    int m_handleOffset;
    
    nutty::DeviceBuffer<Normal> m_normalsSave;

    void ComputeMovement(float dt);

    void UpdateTree(void);

    void DrawText(const char* text, float x, float y);

    void DrawMaterial(Material m_matToPrint)
    {
        static std::stringstream ss;
        ss.str("");
        ss << "\n\nMirror=" << m_matToPrint.isMirror() << " (1)\n";
        ss << "Alpha=" << m_matToPrint.alpha() << " (2,3)\n";
        ss << "Reflectance=" << m_matToPrint.reflectivity() << " (4,5)\n";
        ss << "Fresnel_R=" << m_matToPrint.fresnel_r() << " (6,7)\n";
        ss << "Fresnel_T=" << m_matToPrint.fresnel_t() << " (8,9)\n";
        ss << "IOR=" << m_matToPrint.reflectionIndex() << " (o,p)\n";
        FontBeginDraw();
        FontDrawText(ss.str().c_str(), 0.0025f, 0.5);
        FontEndDraw();
    }

public:
    BaseScene(void) : keyDown_key(0), m_updateTreeEachFrame(1), m_lastTreeBuildTime(0), m_envMapScale(1), m_drawDebugTree(false), m_handleOffset(0), m_drawFont(true), m_animateGeometry(false)
    {

    }

    bool Create(int twidth, int theight, int width, int height);

    CTGeometryHandle AddGeometry(const char* objFile, GeoHandle* geoHandle = NULL);

    virtual void OnRender(float dt);

    void OnResize(int width, int height);

    virtual void OnUpdate(float dt);

    virtual void AfterInit(void) { }

    virtual void OnKeyUp(int key)
    {
        keyDown_key = 0;
    }
    virtual void OnKeyDown(int key);

    virtual void OnMouseMoved(int dx, int dy, int x, int y)
    {
        m_camera.Rotate(dx * 1e-3f, dy * 1e-3f);
    }

    virtual int GetWidth(void) { return m_w; }

    virtual int GetHeight(void) { return m_h; }

    virtual void FillScene(void) = 0;

    virtual ~BaseScene(void);
};

//implementation

bool BaseScene::Create(int twidth, int theight, int width, int height)
{
    m_w = width;
    m_h = height;
    m_tw = twidth;
    m_th = theight;

    assert(FontInit(width, height));

    m_p = glProgram::CreateProgramFromFile("glsl/vs.glsl", "glsl/fs.glsl");
    m_p->Bind();

    int size = twidth * theight * 4;

    m_tb = new glTextureBuffer();

    m_tb->Resize(size * sizeof(float));

    m_tb->BindToTextureSlot(TEXTURE_SLOT_RT_COLOR_BUFFER);

    //checkGLError();

    nutty::Init();

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsGLRegisterBuffer(&m_res, m_tb->BufferId(), 0));

    glUniform1i(glGetUniformLocation(m_p->Id(), "sampler"), TEXTURE_SLOT_RT_COLOR_BUFFER);

    glUniform1i(glGetUniformLocation(m_p->Id(), "twidth"), twidth);

    glUniform1i(glGetUniformLocation(m_p->Id(), "theight"), theight);

    glUniform1i(glGetUniformLocation(m_p->Id(), "width"), width);

    glUniform1i(glGetUniformLocation(m_p->Id(), "height"), height);

    RT_Init(twidth, theight);

    m_atlas = new cuTextureAtlas();
    m_atlas->Init();

    std::string texPath;
    FindFilePath("newhdri_011.jpg", texPath); //
    m_atlas->AddTexture(texPath.c_str());

    CT_SAFE_CALL(CTInit(0));

    uint treeType = CT_CREATE_TREE_GPU;//GPU_DP;
    CT_SAFE_CALL(CTCreateSAHKDTree(&m_tree, treeType));

    m_triGPUData = new GPUTraceData();

    FillScene();

    m_normalsSave.Resize(m_triGPUData->triNormals.Size());
    nutty::Copy(m_normalsSave.Begin(), m_triGPUData->triNormals.Begin(), m_triGPUData->triNormals.End());

    uint vertexCount;
    CT_SAFE_CALL(CTGetPrimitiveCount(m_tree, &vertexCount));
    vertexCount *= 3;

    print("Primitives: %d\n", vertexCount/3);

    m_nodeGPUData = new NodeGPUDataTransformer();

    m_tris.materials = m_triGPUData->materials.Begin()();
    m_tris.normals = m_triGPUData->triNormals.Begin()();
    m_tris.texCoords = m_triGPUData->triTc.Begin()();
    m_tris.matId = m_triGPUData->perVertexMatId.Begin()();

    uint count;
    const cuTextureObj* textures = m_atlas->GetTextures(&count);
    RT_BindTextures(textures, count);

    m_debugLayer = new glDebugLayer();
    m_debugLayer->GetProgram()->Bind();

    GLuint loc = glGetUniformLocation(m_debugLayer->GetProgram()->Id(), "perspective");
    glUniformMatrix4fv(loc, 1, false, (float*)m_camera.GetProjection());
    std::stringstream ss;

    RT_SetViewPort(twidth, theight);

    //m_camera.Move(1, 3, -5);

    chimera::util::Mat4 matrix;
    chimera::util::Mat4 cameraMatrix;
    float time = 0;

    UpdateTree();

    if(treeType == CT_CREATE_TREE_CPU)
    {
        m_debugLayer->ResetGeometry();
        CT_SAFE_CALL(CTTreeDrawDebug(m_tree, m_debugLayer)); //collectdata
    }

    const void* memory = NULL;
    CTuint cnt;
    CT_SAFE_CALL(CTGetRawLinearMemory(m_tree, &cnt, &memory));

    if(treeType == CT_CREATE_TREE_CPU)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc(&m_tris.positions, vertexCount * sizeof(Position)));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_tris.positions, memory, cnt, cudaMemcpyHostToDevice));
    }
    else
    {
        m_tris.positions = (Position*)memory;
    }

    RT_BindGeometry(m_tris);

    timer.VReset();
    traceTimer.VReset();

    OnResize(twidth, theight);

    AfterInit();

    return true;
}

void BaseScene::OnResize(int width, int height)
{
    height = height <= 0 ? 1 : height;
    width = width <= 0 ? 1 : width;
    //glViewport(0, 0, width, height);
    m_camera.ComputeProj(width, height);
    if(m_debugLayer)
    {
        m_debugLayer->GetProgram()->Bind();
        GLuint loc = glGetUniformLocation(m_debugLayer->GetProgram()->Id(), "perspective");
        glUniformMatrix4fv(loc, 1, false, (float*)m_camera.GetProjection());
    }
}

void BaseScene::ComputeMovement(float dt)
{
    float delta = 4*dt;
    if(keyDown_key == KEY_W)
    {
        m_camera.Move(0,0,delta);
    }
    else if(keyDown_key == KEY_S)
    {
        m_camera.Move(0,0,-delta);
    }
    else if(keyDown_key == KEY_D)
    {
        m_camera.Move(delta,0,0);
    }
    else if(keyDown_key == KEY_A)
    {
        m_camera.Move(-delta,0,0);
    }
    else if(keyDown_key == KEY_C)
    {
        m_camera.Move(0,-delta,0);
    }
    else if(keyDown_key == KEY_V)
    {
        m_camera.Move(0,delta,0);
    }
    else if(keyDown_key == KEY_N)
    {
        //g_cameraDistance -= 0.1f;
    }
    else if(keyDown_key == KEY_M)
    {
        //g_cameraDistance += 0.1f;
    }
}

void BaseScene::OnKeyDown(int key)
{
    keyDown_key = key;

    if(key == KEY_M)
    {
        m_drawFont ^= 1;
    }
    else if(key == KEY_P)
    {
        m_updateTreeEachFrame ^= 1;
    }
    else if(key == KEY_L)
    {
        m_animateGeometry ^= 1;
    }
    else if(key == KEY_K)
    {
        RT_IncDepth();
    }
    else if(key == KEY_B)
    {
        m_drawDebugTree ^= true;
    }
    else if(key == KEY_J)
    {
        RT_DecDepth();
    }
    else if(key == KEY_J)
    {
        RT_DecDepth();
    }
    else if(key == KEY_T)
    {
        m_envMapScale -= 0.01f;
        RT_EnvMapSale(m_envMapScale);
    }
    else if(key == KEY_Z)
    {
        m_envMapScale += 0.01f;
        RT_EnvMapSale(m_envMapScale);
    }
    else if(key == KEY_9)
    {
        float4* gpuColors;
        uint h, w;
        gpuColors = RT_GetRawColors(&h, &w);
        float4* hData = (float4*)malloc(w * h * sizeof(float4));
        cudaMemcpy(hData, gpuColors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);
        WritePNGFile(hData, w, h, "test.png");
        free(hData);
    }
}

CTGeometryHandle BaseScene::AddGeometry(const char* objFile, GeoHandle* geoHandle)
{
    RawTriangles* cpuTris;

    chimera::util::HTimer loadTimer;
    auto it = m_modelCache.find(std::string(objFile));
    if(it != m_modelCache.end())
    {
        cpuTris = it->second;
    }
    else
    {
        RawTriangles* _cpuTris = new RawTriangles();
        loadTimer.Start();

        std::string str(objFile);
        if(str[str.size() - 1] == '\k')
        {
            DeSerialize(_cpuTris, objFile);
        }
        else
        {
            std::string modelPath;
            FindFilePath(objFile, modelPath);
            if(!ReadObjFileThreaded(modelPath.c_str(), *_cpuTris))
            {
                print("Couldn't load model!\n");
                return -1;
            }
            m_modelCache.insert(std::pair<std::string, RawTriangles*>(std::string(objFile), _cpuTris));
            Serialize(_cpuTris, objFile);
        }
        cpuTris = _cpuTris;
    }

    loadTimer.Stop();
    print("Loading '%s' took '%f' Seconds (Used Cache '%d')\n", objFile, loadTimer.GetSeconds(), (CTuint)(it != m_modelCache.end()));

    size_t tcstart = m_triGPUData->triTc.Size();
    m_triGPUData->triTc.Resize(tcstart + cpuTris->tcoords.size());

    size_t nstart = m_triGPUData->triNormals.Size();
    m_triGPUData->triNormals.Resize(nstart + cpuTris->normals.size());

    size_t matIdstart = m_triGPUData->perVertexMatId.Size();

    if(!cpuTris->tcoords.empty())
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_triGPUData->triTc.Begin()() + tcstart, &(cpuTris->tcoords[0]), cpuTris->tcoords.size() * sizeof(TexCoord), cudaMemcpyHostToDevice));
    }

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_triGPUData->triNormals.Begin()() + nstart, &(cpuTris->normals[0]), cpuTris->normals.size() * sizeof(Normal), cudaMemcpyHostToDevice));

    std::map<std::string, int> texToSlot;
    for(auto& it = cpuTris->materials.begin(); it !=  cpuTris->materials.end(); ++it)
    {
        if(it->second.texFile.size() && texToSlot.find(it->second.texFile) == texToSlot.end())
        {
            std::string texPath;
            if(FindFilePath(it->second.texFile.c_str(), texPath))
            {
                int slot = m_atlas->AddTexture(texPath.c_str());
                texToSlot[it->second.texFile] = slot;
            }
            else
            {
                print("Texture %s not found\n", it->second.texFile.c_str());
            }
        }
    }

    uint matOffset = (uint)m_triGPUData->materials.GetPos();

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
        m_triGPUData->materials.PushBack(mat);
    }

    CTGeometryHandle handle = 0;

    std::vector<byte> matsIds;

    for(auto& i = cpuTris->intervals.begin(); i != cpuTris->intervals.end(); ++i)
    {
        //matsIds.reserve(matsIds.size() + (i->end - i->start)*3);
        byte matIndex = matOffset + cpuTris->GetMaterialIndex(i->material);

        //matsIds.resize((i->end - i->start)*3);
        for(int k = 0; k < (i->end - i->start)*3; ++k)
        {
            matsIds.push_back(matIndex);
        }
        //matsIds.insert(matsIds.front(), (i->end - i->start)*3, matIndex);

//         ICTGeometry* geo;
//         CTCreateGeometry(&geo);
//         for(int a = i->start; a < i->end; ++a)
//         {
//             CTTriangle tri;
//             for(int k = 0; k < 3; ++k)
//             {
//                 CTreal3 pos;
//                 Position p = cpuTris->positions[3 * a + k];
//                 pos.x = p.x;
//                 pos.y = p.y;
//                 pos.z = p.z;
//                 tri.SetValue(k, pos);
//             }
//             CTAddPrimitive(geo, &tri);
//         }

        CTreal3* ptr = &cpuTris->positions[0];
        CT_SAFE_CALL(CTAddGeometryFromLinearMemory(m_tree, ptr + 3*i->start, 3*(i->end - i->start), &handle));
        if(geoHandle)
        {
            geoHandle->handle = handle;
            geoHandle->start = 3*i->start + m_handleOffset;
            geoHandle->end = 3*i->end + m_handleOffset;
            m_handleOffset += 3*i->end - 3*i->start;
        }
    }

    uint start = (uint)m_triGPUData->perVertexMatId.Size();
    m_triGPUData->perVertexMatId.Resize(m_triGPUData->perVertexMatId.Size() + matsIds.size());
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_triGPUData->perVertexMatId.Begin()() + start, &matsIds[0], matsIds.size() * sizeof(byte), cudaMemcpyHostToDevice));

    return handle;
}

#undef GLOBAL_PROFILE
static uint steps = 0;
static double g_time = 0;

void BaseScene::UpdateTree(void)
{
    CT_TREE_DEVICE type;
    CT_SAFE_CALL(CTGetTreeDeviceType(m_tree, &type));

    chimera::util::HTimer loadTimer;
    {
        print("Building Tree ...\n");
        if(type == eCT_GPU) cudaStreamSynchronize(m_tree->GetStream());
        loadTimer.Start();
        CT_SAFE_CALL(CTUpdate(m_tree));
        if(type == eCT_GPU) cudaStreamSynchronize(m_tree->GetStream());
        loadTimer.Stop();
        m_lastTreeBuildTime = loadTimer.GetMillis();
        print("Building Tree took '%f' Seconds\n", loadTimer.GetSeconds());
#ifdef GLOBAL_PROFILE
        g_time += m_lastTreeBuildTime;
        if(steps++ == 50)
        {
            std::ofstream bench("bench.txt");
            print("Building Tree took Average '%f' Milliseconds\n", g_time/(double)steps);
            bench << g_time/(double)steps << "\n";
            bench.close();
            exit(0);
        }
#endif
    }    
    
    CTuint nodeCount;
    CT_SAFE_CALL(CTGetInteriorNodeCount(m_tree, &nodeCount));

    CTuint leafs;
    CT_SAFE_CALL(CTGetLeafNodeCount(m_tree, &leafs));

    print("Getting Tree Data ...\n");
    loadTimer.Start();

    TreeNodes gpuTree;

    if(type == eCT_CPU)
    {
        m_nodeGPUData->Reset();
        m_nodeGPUData->Resize(leafs, nodeCount);

        m_nodeGPUData->Fill(m_tree, eCT_NODE_IS_LEAF, m_nodeGPUData->lineartreeNodeIsLeaf);
        m_nodeGPUData->Fill(m_tree, eCT_LEAF_NODE_PRIM_START_INDEX, m_nodeGPUData->lineartreeContentStart);
        m_nodeGPUData->Fill(m_tree, eCT_LEAF_NODE_PRIM_COUNT, m_nodeGPUData->lineartreeContentCount);
        m_nodeGPUData->Fill(m_tree, eCT_NODE_LEFT_CHILD, m_nodeGPUData->lineartreeLeftNode);
        m_nodeGPUData->Fill(m_tree, eCT_NODE_RIGHT_CHILD, m_nodeGPUData->lineartreeRightNode);
        m_nodeGPUData->Fill(m_tree, eCT_NODE_SPLITS, m_nodeGPUData->lineartreeSplit);
        m_nodeGPUData->Fill(m_tree, eCT_NODE_SPLIT_AXIS, m_nodeGPUData->lineartreeSplitAxis);
        m_nodeGPUData->Fill(m_tree, eCT_NODE_TO_LEAF_INDEX, m_nodeGPUData->nodeIndexToLeafIndex);

        gpuTree.leafIndex = m_nodeGPUData->nodeIndexToLeafIndex.Begin()();
        gpuTree.contentCount = m_nodeGPUData->lineartreeContentCount.Begin()();
        gpuTree.contentStart = m_nodeGPUData->lineartreeContentStart.Begin()();

        CTuint d;
        CTGetDepth(m_tree, &d);
        if(d > 0)
        {
            gpuTree._left = m_nodeGPUData->lineartreeLeftNode.Begin()();
            gpuTree._right = m_nodeGPUData->lineartreeRightNode.Begin()();
            gpuTree.split = m_nodeGPUData->lineartreeSplit.Begin()();
            gpuTree.splitAxis = m_nodeGPUData->lineartreeSplitAxis.Begin()();
        }

        gpuTree.isLeaf = m_nodeGPUData->lineartreeNodeIsLeaf.Begin()();

        uint cnt;
        const void* memory;
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, &memory, eCT_LEAF_NODE_PRIM_IDS));
        m_nodeGPUData->lineartreeContent.Resize(cnt / sizeof(uint));
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpyAsync(m_nodeGPUData->lineartreeContent.Begin()(), memory, cnt, cudaMemcpyHostToDevice, m_tree->GetStream()));

        gpuTree.content = m_nodeGPUData->lineartreeContent.Begin()();
    }
    else if(type == eCT_GPU)
    {
        CTuint cnt;
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree.isLeaf, eCT_NODE_IS_LEAF));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree.contentStart, eCT_LEAF_NODE_PRIM_START_INDEX));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree.contentCount, eCT_LEAF_NODE_PRIM_COUNT));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree._left, eCT_NODE_LEFT_CHILD));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree._right, eCT_NODE_RIGHT_CHILD));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree.split, eCT_NODE_SPLITS));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree.splitAxis, eCT_NODE_SPLIT_AXIS));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree.leafIndex, eCT_NODE_TO_LEAF_INDEX));
        CT_SAFE_CALL(CTGetLinearMemory(m_tree, &cnt, (const void**)&gpuTree.content, eCT_LEAF_NODE_PRIM_IDS));
    }

    loadTimer.Stop();
    print("Traversing Tree took '%f' Seconds\n", loadTimer.GetSeconds());

    RT_BindTree(gpuTree);
}

void BaseScene::OnUpdate(float dt)
{
    if(m_updateTreeEachFrame)
    {
        UpdateTree();

        if(m_tree->GetDeviceType() == CT_CREATE_TREE_CPU)
        {
            const void* memory = NULL; 
            CTuint cnt;
            CT_SAFE_CALL(CTGetRawLinearMemory(m_tree, &cnt, &memory));
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(m_tris.positions, memory, cnt, cudaMemcpyHostToDevice));
        }
        cudaStreamSynchronize(m_tree->GetStream());
    }

    ComputeMovement(dt);
}

void BaseScene::OnRender(float dt)
{
    static int frame = 0;

    int width = m_w;
    int height = m_h;

    int twidth = m_tw; int theight = m_th;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glViewport(twidth, 0, width - twidth, height);

    int glError = glGetError();
    if(glError != GL_NO_ERROR)
    {
        print("%d\n", glError);
    }

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsMapResources(1, &m_res));

    float4* mappedPtr;
    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsResourceGetMappedPointer((void**)&mappedPtr, NULL, m_res));

    const ICTAABB* aabb;
    CT_SAFE_CALL(CTGetAxisAlignedBB(m_tree, &aabb));
    BBox bbox;
    bbox.init();
    bbox.addPoint(aabb->GetMin());
    bbox.addPoint(aabb->GetMax());

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSynchronize());
    traceTimer.Start();
    RT_Trace(mappedPtr, m_camera.GetView(), m_camera.GetEye(), bbox);

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnmapResources(1, &m_res));

    CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSynchronize());

    traceTimer.Stop();

    m_p->Bind();
    //glViewport(0, 0, twidth, theight);
    glViewport(0, 0, width, height);
    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    if(m_drawDebugTree)
    {
        m_debugLayer->BeginDraw();
        GLuint loc = glGetUniformLocation(m_debugLayer->GetProgram()->Id(), "view");
        glUniformMatrix4fv(loc, 1, false, (float*)m_camera.GetIView());
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        m_debugLayer->ResetGeometry();
        CT_SAFE_CALL(CTTreeDrawDebug(m_tree, m_debugLayer));
        m_debugLayer->DrawGLGeo();
        glDisable(GL_BLEND);
        m_debugLayer->EndDraw();
    }

    if(m_drawFont)
    {
        float xoffset = 0;// m_tw / (float)m_w;
        //font
        uint inc;
        uint lnc;
        CTGetInteriorNodeCount(m_tree, &inc);
        CTGetLeafNodeCount(m_tree, &lnc);
        std::stringstream ss;
        ss.str("");
        ss << "Nodes: " << (inc + lnc) << "\n";
        ss << "Leafes: " << lnc << "\n";
        ss << "Depth: " << m_tree->GetDepth() << "\n";
        double traceMillis = traceTimer.GetMillis();
        
        if(!m_updateTreeEachFrame)
        {
            m_lastTreeBuildTime = 0;
        }

        double allMillis = (double)(traceMillis + m_lastTreeBuildTime);
        ss << "Frame: " << frame << "\n";
        ss << twidth * theight << " Pixel\n";
        ss << RT_GetLastRayCount() << " Rays\n";
        ss << m_tree->GetPrimitiveCount() << " Primitives\n";
        //ss << (int)(1000.0 / (traverseTimer.GetMillis() == 0 ? 1 : traverseTimer.GetMillis())) << " FPS (Build)\n";
        ss << "Tracing: " << traceMillis <<" ms\n";
        ss << "Building: " << m_lastTreeBuildTime << " ms\n";
        ss << "Overall Time: " << allMillis << " ms\n";

        //     ss << "\n\nMirror=" << g_matToPrint.isMirror() << " (1)\n";
        //     ss << "Alpha=" << g_matToPrint.alpha() << " (2,3)\n";
        //     ss << "Reflectance=" << g_matToPrint.reflectivity() << " (4,5)\n";
        //     ss << "Fresnel_R=" << g_matToPrint.fresnel_r() << " (6,7)\n";
        //     ss << "Fresnel_T=" << g_matToPrint.fresnel_t() << " (8,9)\n";
        //     ss << "IOR=" << g_matToPrint.reflectionIndex() << " (o,p)\n";

        FontBeginDraw();
        //     std::string info;
        //     RT_GetRayInfo(info);
        //     ss << info << "\n";

        ss << "\n\nTracedepth (-j, +k) = " << RT_GetRecDepth() << "\n";
        //     ss << "Animate Camera (g) = " << g_animateCamera << "\n";
        //     ss << "Animate Light (h) = " << g_animateLight << "\n";

        FontDrawText(ss.str().c_str(), xoffset + 0.0025f, 0);
        FontEndDraw();

        //font end
    }

    frame++;
}

BaseScene::~BaseScene(void)
{
    for(auto& it = m_modelCache.begin(); it != m_modelCache.end(); ++it)
    {
        delete it->second;
    }

    delete m_debugLayer;
    delete m_triGPUData;
    delete m_nodeGPUData;

    RT_Destroy();

    delete m_atlas;
    cudaFree(m_tris.materials);

    if(m_tree->GetDeviceType() == eCT_CPU)
    {
        cudaFree(m_tris.positions);
    }

    cudaFree(m_tris.normals);
    cudaFree(m_tris.texCoords);
    cudaFree(m_tris.matId);

    cudaGraphicsUnregisterResource(m_res);

    delete m_p;
    delete m_tb;

    CT_SAFE_CALL(CTRelease());
    nutty::Release();

    FontDestroy();
}