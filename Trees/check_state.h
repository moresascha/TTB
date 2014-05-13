#pragma once
#include "ct.h"

#ifdef _DEBUG
#define checkState(state) if(! ( state ) ) { return CT_INVALID_OPERATION; }
#else
#define checkState(state) { do {} while (0); }
#endif