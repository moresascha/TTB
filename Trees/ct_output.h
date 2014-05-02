#pragma once

extern void __ct_printf(char* format, ...);

#define ct_printf __ct_printf