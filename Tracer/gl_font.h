#pragma once

bool FontInit(unsigned int width, unsigned int height);

void FontBeginDraw(void);

void FontEndDraw(void);

void FontDrawText(const char* text, float x, float y);

void FontDestroy(void);
