#pragma once
#include "ct.h"

#define checkState(state) if(! ( state ) ) { return CT_INVALID_OPERATION; }