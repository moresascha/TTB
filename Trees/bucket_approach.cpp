#include "ct_runtime.h"
#include <Nutty.h>
#include "ct_output.h"
#include <HostBuffer.h>
#include <Copy.h>
#include <Fill.h>
#include <Functions.h>

void CreateBox(ICTGeometry** geo, float midx, float midy)
{
    CT_SAFE_CALL(CTCreateGeometry(geo));

    ICTVertex* v;
    CT_SAFE_CALL(CTCreateVertex(&v));
    ctfloat3 pos;
    pos.x = midx - 1;
    pos.y = midy - 1;
    pos.z = 0;
    v->SetPosition(pos);
    (*geo)->AddVertex(v);

    pos.x = midx + 1;
    pos.y = midy - 1;
    pos.z = 0;
    v->SetPosition(pos);
    (*geo)->AddVertex(v);

    pos.x = midx - 1;
    pos.y = midy + 1;
    pos.z = 0;
    v->SetPosition(pos);
    (*geo)->AddVertex(v);

    pos.x = midx + 1;
    pos.y = midy + 1;
    pos.z = 0;
    v->SetPosition(pos);
    (*geo)->AddVertex(v);
}

template <typename T>
void printBinary(T i)
{
    int bytes = sizeof(T);
    int bits = bytes * 8;

    for(int b = bits - 1; b >= 0; --b)
    {
        byte c = ((i & (1 << b)) == (1 << b)) ? '1' : '0';
        ct_printf("%c", c);
    }
    ct_printf("\n");
}


void insertToBucket(int v, nutty::HostBuffer<int>& buckets)
{

}

extern "C" void bucketTest(void)
{
    nutty::HostBuffer<float> vals(10);

    nutty::Fill(vals.Begin(), vals.End(), nutty::unary::RandNorm<float>());

    nutty::HostBuffer<int> buckets(100);

    float f = 4.004;
    float prec = 1000.0f;
    int ipart = (int)f;
    int fpart = (int) (prec * (f - (int)f));
    ct_printf("%d, %d\n", ipart, fpart);
    printBinary(ipart);
    printBinary(fpart);
}