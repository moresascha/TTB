#include <windows.h>
#include <windowsx.h>
#include "window.h"

#include "scene.h"
#include "gl_layer.h"

#include <chimera/api/ChimeraAPI.h>
#include <chimera/Timer.h>

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

WPARAM g_key;
unsigned int g_lastX = 0;
unsigned int g_lastY = 0;

IScene* g_currentScene;

bool g_bEnd = false;

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
        g_currentScene->OnResize(LOWORD(lParam),HIWORD(lParam));
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

                g_currentScene->OnMouseMoved(dx, dy, x, y);

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
            g_currentScene->OnKeyUp(g_key);
            return 0;
        }
    case WM_KEYDOWN :
        {
            g_key = wParam;
            if(g_key == KEY_ESC)
            {
                g_bEnd = true;
                break;
            }
            g_currentScene->OnKeyDown(g_key);
            return 0;
        } break;
    default:
        break;
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR str, int nCmdShow)
{

// #ifdef _DEBUG
// #define _CRTDBG_MAP_ALLOC
//     _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_CHECK_CRT_DF | _CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_EVERY_16_DF);
// #endif
    int glWindowsWidth = 1024;
    int glWindowsHeight = 768;
    int width = 1024;//2048+768+64;//(1024 * 3) / 2;
    int height = 768;//2048+768+64;//(512 * 3) / 2;
    TCHAR* clazz = L"TC";
    HWND hwnd = CreateScreen(hInstance, WndProc, clazz, L"", glWindowsWidth, glWindowsHeight);

    RECT rect;
    GetClientRect(hwnd, &rect);
    setlocale(LC_ALL, "German_Germany");
    HGLRC context; 
    if(!(context = CreateGLContextAndMakeCurrent(hwnd)))
    {
        return -1;
    }
    //SetWindowLong(hwnd, GWL_STYLE, 0);

    g_currentScene = RT_CreateExampleScene(width, height, glWindowsWidth, glWindowsHeight);

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);  

    HDC hDC = GetDC(hwnd);

    glClearColor(0,0,0,0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SwapBuffers(hDC);

    //g_currentScene->OnResize(width, height);

    chimera::util::HTimer timer;
    timer.VReset();
    MSG msg;
    while(!g_bEnd)
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
            g_currentScene->OnUpdate(1e-3f*timer.VGetLastMillis());
            g_currentScene->OnRender(1e-3f*timer.VGetLastMillis());
            SwapBuffers(hDC);

//             float3 vp;
//             if(g_animateLight)
//             {
//                 v.Set(0, 20, -20);
// 
//                 //matrix.RotateX(0.25*dt);
//                 matrix.RotateY(dt);
// 
//                 v = chimera::util::Mat4::Transform(matrix, v);
// 
//                 vp.x = v.GetX();
//                 vp.y = v.GetY();
//                 vp.z = v.GetZ();
// 
//                 RT_SetLightPos(vp);
//             }
// 
//             if(g_animateCamera)
//             {
//                 float3 lookAt;
//                 lookAt.x = 0;
//                 lookAt.y = 3;
//                 lookAt.z = 0;
// 
//                 v.Set(0, 4, -g_cameraDistance);
//                 cameraMatrix.RotateY(-0.25f*dt);
// 
//                 v = chimera::util::Mat4::Transform(cameraMatrix, v);
//                 vp.x = v.GetX();
//                 vp.y = v.GetY();
//                 vp.z = v.GetZ();
//                 g_cam.LookAt(lookAt, vp);
//             }
            timer.VTick();
        }
    }
    ShowWindow(hwnd, SW_HIDE);

    delete g_currentScene;

    ReleaseGLContext(context);

    ReleaseScreen(hInstance, hwnd, clazz);

    return 0;
}