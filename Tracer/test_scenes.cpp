#include "scene.cpp"

#define INIT_LP 5,12,-5

class TestObjScene : public BaseScene
{
private:
    GeoHandle m_handle;
    float rotateX;
    float rotateY;
    const char* m_obj;

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
        Material mat = m_triGPUData->materials[1];
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
        static uint d = 25;
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
        else if(key == KEY_6)
        {
            m_tree->SetHCDepth(d++);
        }
        else if(key == KEY_5)
        {
            m_tree->SetHCDepth(--d);
        }

        BaseScene::OnKeyDown(key);
    }

    void OnRender(float dt)
    {
        BaseScene::OnRender(dt);
        if(m_drawFont)
        {
            DrawMaterial(m_matToPrint);
        }
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
    CTGeometryHandle m_hBunny;
    GeoHandle m_hGeoBunny;
public:
    Room(void) : m_animateLight(false)
    {
        m_camera.SetEyePos(make_float3(5.367411, 5.367411, -15));
        m_animateGeometry = true;
    }

    void OnKeyDown(int key)
    {
        if(key == KEY_H)
        {
            m_animateLight ^= true;
        }
        else if(key <= KEY_1 || key <= KEY_6)
        {
            RT_SetShader(key - 0x31);
        }
        uint matSlot = 2;
        Material mat = m_triGPUData->materials[matSlot];
        // 
        //Alpha
        if(key == 98)
        {
            mat._alpha -= 0.01f;
            mat._alpha = 0.5f;
        }
        else if(key == 99)
        {
            mat._alpha += 0.01f;
            mat._alpha = 1.5f;
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
        m_triGPUData->materials.Insert(matSlot, mat);

        BaseScene::OnKeyDown(key);
    }

    void TransformFlyingStuff(float time, bool init = false)
    {
        if(!m_updateTreeEachFrame) return;

        chimera::util::cmRNG rng(3);
        
//         chimera::util::Mat4 model;
//         model.RotateY(time);
//         CT_SAFE_CALL(CTTransformGeometryHandle(m_tree, m_hBunny, (CTreal4*)model.m_m.m));
//         if(!init)
//         {
//             RT_TransformNormals(m_normalsSave.Begin()(), m_tris.normals, (CTreal4*)model.m_m.m, m_hGeoBunny.start, (m_hGeoBunny.end - m_hGeoBunny.start), m_tree->GetStream());
//         }
        static float ttime = 0;
        int i = 0;
        for(auto it = flyingStuff.begin(); it != flyingStuff.end(); ++it, i++)
        {
            chimera::util::Mat4 model;
//             int offset = 0;
//             int scale = 10;
//             chimera::util::Vec3 t(offset + rng.NextCubeFloat(scale), 5 + rng.NextCubeFloat(4), offset + rng.NextCubeFloat(scale));
//             chimera::util::Mat4 tm;
//             tm.RotateY((rng.NextInt()%2==0 ? -1 : 1) * time);
//             t = chimera::util::Mat4::Transform(tm, t);
//             model.SetTranslation(t);
//             model.Scale(1.5f);
            float scale = 9;
            int count = flyingStuff.size();
            float phi = ttime + 2*XM_PI * i / (float)(count-1);
            model.SetTranslation(scale * sin(phi), 2, scale * cos(phi));
            model.SetScale(1.2f);
            CT_SAFE_CALL(CTTransformGeometryHandle(m_tree, it->handle, (CTreal4*)model.m_m.m));
            if(!init)
            {
                RT_TransformNormals(m_normalsSave.Begin()(), m_tris.normals, (CTreal4*)model.m_m.m, it->start, (it->end - it->start), m_tree->GetStream());
            }
        }
        ttime += 16 * 1e-3f;
    }

    void OnUpdate(float dt)
    {
        static float time = 0;
        static float lightTime = 0;
        //m_updateTreeEachFrame = true;
        if(m_animateGeometry)
        {
            TransformFlyingStuff(time);
        }

        for(int i = 0; i < 1 && m_animateLight; ++i)
        {
            chimera::util::Mat4 matrix;
            matrix.SetRotateY(lightTime);

            chimera::util::Vec3 v = chimera::util::Mat4::Transform(matrix, chimera::util::Vec3(INIT_LP));
            light[i]->SetPosition(make_float3(v.x, v.y, v.z));

            lightTime += dt;
        }

        time += dt * 0.5f;

        BaseScene::OnUpdate(dt);
    }

    void OnRender(float dt)
    {
        BaseScene::OnRender(dt);
        if(m_drawFont)
        {
            DrawMaterial(m_matToPrint);
        }
    }

    bool m_animateObj;
    std::vector<GeoHandle> flyingStuff;

    void FillScene(void)
    {
        AddGeometry("empty_room_big_bunny.obj");
        m_hBunny = AddGeometry("bunny.obj", &m_hGeoBunny);
        m_animateGeometry = false;

        GeoHandle hhandle;

        int count = 16;
        float scale = 9;
        chimera::util::Mat4 model;
        for(int i = 0; i < count; ++i)
        {
            float phi = 2*XM_PI * i / (float)(count-1);
            CTGeometryHandle handle = AddGeometry("notc_sphere.obj", &hhandle);
            model.SetTranslation(scale * sin(phi), 2, scale * cos(phi));
            model.SetScale(1.2f);
            CT_SAFE_CALL(CTTransformGeometryHandle(m_tree, handle, (CTreal4*)model.m_m.m));
            flyingStuff.push_back(hhandle);
        }

        m_updateTreeEachFrame = false;
        TransformFlyingStuff(1, true);
        m_updateTreeEachFrame = true;

        RT_AddLight(&light[0]);
        light[0]->SetColor(make_float3(1,1,1));
        light[0]->SetIntensity(1);
        light[0]->SetPosition(make_float3(INIT_LP));
        light[0]->SetRadius(60);
    }
};

class BMW : public BaseScene
{
private:
    RT_Light_t light[2];
    bool m_animateLight;
    float3 lightColor;
    bool m_animateCamera;
public:
    BMW(void) : m_animateLight(false)
    {
        m_camera.SetEyePos(make_float3(5.924796, 2.040130, -4.645228));
        //m_camera.LookAt(make_float3(0,5,0), make_float3(0, 2, 10));
        m_updateTreeEachFrame = false;
        m_animateCamera = false;
    }

    void OnUpdate(float dt)
    {
        static float time = 0;
        static float lightTime = 0;

        for(int i = 0; i < 1 && m_animateLight; ++i)
        {
            chimera::util::Mat4 matrix;
            matrix.SetRotateY(lightTime);

            chimera::util::Vec3 v = chimera::util::Mat4::Transform(matrix, chimera::util::Vec3(INIT_LP));
            light[i]->SetPosition(make_float3(v.x, v.y, v.z));
            //print("Pos: %f %f %f\n", v.x, v.y, v.z);
            lightTime += dt;
        }
        //print("Dir: %f %f %f\n", m_camera.().x, m_camera.GetEye().x, m_camera.GetEye().x);

        if(m_animateCamera)
        {
            chimera::util::Vec3 eye(5.924796, 2.040130, -4.645228);
            
            chimera::util::Mat4 matrix;
            matrix.SetRotateY(time);
            eye = chimera::util::Mat4::Transform(matrix, eye);
            m_camera.LookAt(make_float3(0,2,0), make_float3(eye.x, eye.y, eye.z));
            time += 0.5*dt;
        }
        BaseScene::OnUpdate(dt);
    }

    GeoHandle hh;
    CTGeometryHandle ctHandle;
//     void AfterInit(void)
//     {
//         chimera::util::Mat4 model;
//         model.RotateY(0.5f);
//         //model.Scale(0.1f);
//         CT_SAFE_CALL(CTTransformGeometryHandle(m_tree, ctHandle, (CTreal4*)model.m_m.m));
//         RT_TransformNormals(m_normalsSave.Begin()(), m_tris.normals, (CTreal4*)model.m_m.m, hh.start, (hh.end - hh.start), m_tree->GetStream());
//         UpdateTree();
//     }

    void OnKeyDown(int key)
    {
        if(key == KEY_H)
        {
            m_animateLight ^= true;
        }
        else if(key <= KEY_1 || key <= KEY_6)
        {
            RT_SetShader(key - 0x31);
        } else if(key == KEY_T)
        {
            lightColor += make_float3(0.1,0.1,0.1);
            light[0]->SetColor(lightColor);
        } else if(key == KEY_Z)
        {
            lightColor -= make_float3(0.1,0.1,0.1);
            light[0]->SetColor(lightColor);
        }
        else if(key == KEY_G)
        {
            m_animateCamera ^= true;
        }

        BaseScene::OnKeyDown(key);
    }

    void FillScene(void)
    {
        AddGeometry("empty_room_bigBMW.obj");
        ctHandle = AddGeometry("mikepan_bmw3v3.obj", &hh);

        lightColor = make_float3(1,1,1);
        RT_AddLight(&light[0]);
        light[0]->SetColor(make_float3(1,1,1)); //make_float3(0.3f,1.0f,0.3f));
        light[0]->SetIntensity(0.8f);
        light[0]->SetPosition(make_float3(INIT_LP));
        light[0]->SetRadius(60);

        /*RT_AddLight(&light[1]);
        light[1]->SetColor(make_float3(1,1,1)); //make_float3(0.3f,1.0f,0.3f));
        light[1]->SetIntensity(0.3f);
        light[1]->SetPosition(make_float3(5.924796, 2.040130, 2.645228));
        light[1]->SetRadius(60);*/
    }
};


IScene* RT_CreateExampleScene(int traceWidth, int traceHeight, int screenW, int screenH)
{
    const char* scenes[] = 
    { 
        "Spiral_Caged.obj", //0
        "mikepan_bmw3v3.obj", //1
        "dragoni.obj",  //2
        "angel.obj",  //3
        "bunny.obj", //4
        "notc_sphere.obj", //5
        "sponza.obj", //6
        "happyi.obj", //7
        "dragoni.obj.idk", //8
        "bunny.obj.idk", //9
        "happyi.obj.idk", //10
        "angel.obj.idk" //11
    };

    BaseScene* scene;
    //scene = new TestObjScene("angel.obj.idk");
    scene = new Room();
    //scene = new BMW();

//     int width = (1024 * 3) / 2;
//     int height = (512 * 3) / 2;

    scene->Create(traceWidth, traceHeight, screenW, screenH);
    return scene;
}