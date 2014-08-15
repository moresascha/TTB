#pragma once

class IScene
{
public:
    virtual void OnUpdate(float dt) = 0;
    virtual void OnResize(int width, int height) = 0;
    virtual void OnRender(float dt) = 0;
    virtual void OnKeyUp(int key) = 0;
    virtual void OnKeyDown(int key) = 0;
    //virtual void OnMouseButtonPressed(int key) {}
    virtual void OnMouseMoved(int dx, int dy, int x, int y) = 0;
    virtual int GetWidth(void) = 0;
    virtual int GetHeight(void) = 0;

    virtual ~IScene(void) { }
};

IScene* RT_CreateExampleScene(void);