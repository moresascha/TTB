#include "scene.cpp"

class TestObjScene : public BaseScene
{
private:
    GeoHandle m_handle;
    float rotateX;
    float rotateY;
    const char* m_obj;
    Material m_matToPrint;

    void TransformObj(void)
    {
        chimera::util::Mat4 model;
        model.RotateX(rotateX);
        model.RotateY(rotateY);
        CT_SAFE_CALL(CTTransformGeometryHandle(m_tree, m_handle.handle, (CTreal4*)model.m_m.m));
        RT_TransformNormals(m_normalsSave.Begin()(), m_tris.normals, (CTreal4*)model.m_m.m, m_handle.start, (m_handle.end - m_handle.start), m_tree->GetStream());
        UpdateTree();
        cudaDeviceSynchronize();
    }

public:
    TestObjScene(const char* obj) : m_obj(obj), rotateX(0), rotateY(0)
    {
        m_updateTreeEachFrame = true;
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

        BaseScene::OnKeyDown(key);
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
    }
};

class Room : public BaseScene
{
private:
    RT_Light_t light[2];
    bool m_animateLight;
public:
    Room(void) : m_animateLight(false)
    {
        m_camera.Move(0, 5, -2);
    }

    void OnKeyDown(int key)
    {
        if(key == KEY_H)
        {
            m_animateLight ^= true;
        }
        else if(key == KEY_1)
        {
            RT_SetShader(0);
        }
        else if(key == KEY_2)
        {
            RT_SetShader(1);
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
/*            RT_TransformNormals(m_normalsSave.Begin()(), m_tris.normals, (CTreal4*)model.m_m.m, it->start, (it->end - it->start), m_tree->GetStream());*/
        }
    }

    void OnUpdate(float dt)
    {
        static float time = 0;
        static float lightTime = 0;

        //TransformFlyingStuff(time);
        //m_updateTreeEachFrame = false;
        
        for(int i = 0; i < 1 && m_animateLight; ++i)
        {
            chimera::util::Mat4 matrix;
            matrix.SetRotateY(lightTime);

            chimera::util::Vec3 v = chimera::util::Mat4::Transform(matrix, chimera::util::Vec3(5, 20, -3));
            light[i]->SetPosition(make_float3(v.x, v.y, v.z));

            lightTime += dt;
        }

        time += dt * 0.5f;

        BaseScene::OnUpdate(dt);
    }

    bool m_animateObj;
    std::vector<GeoHandle> flyingStuff;

    void FillScene(void)
    {
        //AddGeometry("empty_room_big.obj");
        AddGeometry("bunny.obj");
        //AddGeometry("sponza.obj");

        GeoHandle hhandle;
//         chimera::util::cmRNG rng;
        for(int i = 0; i < 0; ++i)
        {
            CTGeometryHandle handle = AddGeometry("cube.obj", &hhandle);

            flyingStuff.push_back(hhandle);
        }

        TransformFlyingStuff(1);

        RT_AddLight(&light[0]);
        light[0]->SetColor(make_float3(0.3f,1.0f,0.3f));
        light[0]->SetIntensity(0.3f);
        light[0]->SetPosition(make_float3(5,20,-3));
        light[0]->SetRadius(60);

//         RT_AddLight(&light[1]);
//         light[1]->SetColor(make_float3(0.2f,0.2,0));
//         light[1]->SetPosition(make_float3(-10,10,10));
//         light[1]->SetRadius(100);
    }
};

IScene* RT_CreateExampleScene(int screenW, int screenH)
{
    const char* scenes[] = 
    { 
        "Spiral_Caged.obj", 
        "mikepan_bmw3v3.obj", 
        "dragon.obj", 
        "angel.obj", 
        "bunny.obj",
        "notc_sphere.obj",
        "sponza.obj"
    };

    //BaseScene* scene = new TestObjScene(scenes[4]);
    BaseScene* scene = new Room();

//     int width = (1024 * 3) / 2;
//     int height = (512 * 3) / 2;

    scene->Create(screenW, screenH);
    return scene;
}