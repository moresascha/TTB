#pragma once

bool FontInit(void);

void FontBeginDraw(void);

void FontEndDraw(void);

void FontDrawText(const char* text, float x, float y);

void FontDestroy(void);
