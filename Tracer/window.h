#pragma once
#include <windows.h>

extern "C" HWND CreateScreen(HINSTANCE hInstance, WNDPROC WndProc, TCHAR szWindowClass[], TCHAR szTitle[], int width, int height);

extern "C" void ReleaseScreen(HINSTANCE hInstance, HWND hWnd, TCHAR szWindowClass[]);