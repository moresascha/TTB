#include "scene.cpp"

class TestObjScene : public BaseScene
{
private:
    GeoHandle m_handle;
    float rotateX;
    float rotateY;
    const char* m_obj;
    Material m_matToPrint;
    bool m_animateObj;
    std::vector<GeoHandle> flyingStuff;

    void TransformObj(void)
    {
        chimera::util::Mat4 model;
        model.RotateX(rotateX);
        model.RotateY(rotateY);
        CT_SAFE_CALL(CTTransformGeometryHandle(m_tree, m_handle.handle, (CTreal4*)model.m_m.m));
        RT_TransformNormals(m_normalsSave.Begin()(), m_tris.normals, (CTreal4*)model.m_m.m, m_handle.start, (m_handle.end - m_handle.start), m_tree->GetStream());
        //UpdateTree();
        cudaDeviceSynchronize();
    }

public:
    TestObjScene(const char* obj) : m_obj(obj), rotateX(0), rotateY(0), m_animateObj(false)
    {

    }

    void OnKeyDown(int key)
    {
        Material mat = m_triGPUData->materials[0];
        // 
        //Alpha
        if(key == 98)
        {
            mat._alpha -= 0.01f;
        }
        else if(key == 99)
        {
            mat._alpha += 0.01f;
        }

        else if(key == 97)
        {
            mat._mirror ^= 1;
        }

        //Reflectance
        else if(key == 100)
        {
            mat._reflectivity -= 0.01f;
        }
        else if(key == 101)
        {
            mat._reflectivity += 0.01f;
        }

        //Fresnel_R
        else if(key == 102)
        {
            mat._fresnel_r -= 0.01f;
        }
        else if(key == 103)
        {
            mat._fresnel_r += 0.01f;
        }

        //Fresnel_T
        else if(key == 104)
        {
            mat._fresnel_t -= 0.01f;
        }
        else if(key == 105)
        {
            mat._fresnel_t += 0.01f;
        }

        //IOR
        else if(key == KEY_I)
        {
            mat._reflectionIndex -= 0.01f;
        }
        else if(key == KEY_O)
        {
            mat._reflectionIndex += 0.01f;
        }

        m_matToPrint = mat;
        m_triGPUData->materials.Insert(0, mat);
        //CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(gpuMats, (void*)&cpuMats[0], sizeof(Material), cudaMemcpyHostToDevice));

        if(key == KEY_ARROW_UP)
        {
            //g_cameraDistance += 0.1f;
            rotateX += 0.1f;
            TransformObj();
        }
        else if(key == KEY_ARROW_DOWN)
        {
            //g_cameraDistance += 0.1f;
            rotateX -= 0.1f;
            TransformObj();
        }
        else if(key == KEY_ARROW_LEFT)
        {
            //g_cameraDistance += 0.1f;
            rotateY += 0.1f;
            TransformObj();
        }
        else if(key == KEY_ARROW_RIGHT)
        {
            //g_cameraDistance += 0.1f;
            rotateY -= 0.1f;
            TransformObj();
        }

        else if(key == KEY_P)
        {
            m_animateObj ^= 1;
        }

        BaseScene::OnKeyDown(key);
    }

    void TransformFlyingStuff(float time)
    {
        chimera::util::cmRNG rng(3);

        for(auto it = flyingStuff.begin(); it != flyingStuff.end(); ++it)
        {
            chimera::util::Mat4 model;
            int offset = 0;
            int scale = 6;
            chimera::util::Vec3 t(offset + rng.NextCubeFloat(scale), 2*scale + rng.NextCubeFloat(scale), offset + rng.NextCubeFloat(scale));
            chimera::util::Mat4 tm;
            tm.RotateY((rng.NextInt()%2==0 ? -1 : 1) * time);
            t = chimera::util::Mat4::Transform(tm, t);
            model.SetTranslation(t);
            //model.Scale(0.1f);

            CT_SAFE_CALL(CTTransformGeometryHandle(m_tree, it->handle, (CTreal4*)model.m_m.m));
            RT_TransformNormals(m_normalsSave.Begin()(), m_tris.normals, (CTreal4*)model.m_m.m, it->start, (it->end - it->start), m_tree->GetStream());
        }
    }

    void OnUpdate(float dt)
    {
        static float time = 0;

        if(time == 0)
        {
            TransformFlyingStuff(0);
        }

        if(m_animateObj)
        {
            TransformFlyingStuff(time);
            rotateY  += 0.025f;
            TransformObj();
            m_updateTreeEachFrame = true;
            time += dt * 0.5f;
        }
        BaseScene::OnUpdate(dt);
    }

    void OnRender(float dt)
    {
        static std::stringstream ss;
        ss.str("");
        ss << "\n\nMirror=" << m_matToPrint.isMirror() << " (1)\n";
        ss << "Alpha=" << m_matToPrint.alpha() << " (2,3)\n";
        ss << "Reflectance=" << m_matToPrint.reflectivity() << " (4,5)\n";
        ss << "Fresnel_R=" << m_matToPrint.fresnel_r() << " (6,7)\n";
        ss << "Fresnel_T=" << m_matToPrint.fresnel_t() << " (8,9)\n";
        ss << "IOR=" << m_matToPrint.reflectionIndex() << " (o,p)\n";

        BaseScene::OnRender(dt);

        FontBeginDraw();
        FontDrawText(ss.str().c_str(), 0.0025f, 0.5);
        FontEndDraw();
    }

    void FillScene(void)
    {
        AddGeometry(m_obj, &m_handle);

        GeoHandle hhandle;
        chimera::util::Mat4 model;

        chimera::util::cmRNG rng;
        for(int i = 0; i < 4; ++i)
        {
            CTGeometryHandle handle = AddGeometry("notc_sphere.obj", &hhandle);

            flyingStuff.push_back(hhandle);
        }
    }
};

IScene* RT_CreateExampleScene(void)
{
    const char* scenes[] = 
    { 
        "Spiral_Caged.obj", 
        "mikepan_bmw3v3.obj", 
        "dragon.obj", 
        "angel.obj", 
        "bunny.obj",
        "notc_sphere.obj"
    };

    BaseScene* scene = new TestObjScene(scenes[4]);
    scene->Create(768, 768);
    return scene;
}