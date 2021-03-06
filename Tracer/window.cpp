#include "window.h"
#include <stdlib.h>
#include <tchar.h>
#include <string.h>

extern "C" HWND CreateScreen(HINSTANCE hInstance, WNDPROC WndProc, TCHAR szWindowClass[], TCHAR szTitle[], int width, int height)
{
    DWORD style = WS_OVERLAPPEDWINDOW;
    RECT rec = {0,0, width, height};
    AdjustWindowRect(&rec, style, false);

    width = rec.right - rec.left;
    height = rec.bottom - rec.top;

    WNDCLASSEX wcex = {0};
    wcex.cbSize = sizeof(wcex);
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = hInstance;
    wcex.lpszClassName = szWindowClass;

    if(!RegisterClassEx(&wcex))
    {
        return 0;
    }

    HWND hwnd = CreateWindowEx(
        0, 
        szWindowClass, 
        szTitle, 
        style,
        0, 
        0, 
        width, 
        height, 
        NULL, 
        NULL,
        hInstance, 
        NULL);

    if(!hwnd)
    {
        return 0;
    }

    return hwnd;
}

extern "C" void ReleaseScreen(HINSTANCE hInstance, HWND hWnd, TCHAR szWindowClass[])
{
    DestroyWindow(hWnd);
    UnregisterClass(szWindowClass, hInstance);
}