#include <stdio.h>
#include <stdlib.h>  
#include "cuKDtree.h"
#include <chimera/Timer.h>
// 
// __global__ void compactionTest(
//     CTuint N,
//     CTbyte srcIndex)
// {
//     RETURN_IF_OOB(N);
//     EVENT_TRIPLE_HEADER_SRC;
//     EVENT_TRIPLE_HEADER_DST;
// 
//     CTuint masks[3];
//     masks[0] = cms[0].mask[id];
//     masks[1] = cms[1].mask[id];
//     masks[2] = cms[2].mask[id];
// 
//     #pragma unroll
//     for(CTaxis_t i = 0; i < 3; ++i)
//     {
//         if(isSet(masks[i]))
//         {
//             CTuint eventIndex = cms[i].index[id];
// 
//             CTuint dstAdd = cms[i].scanned[id];
// 
//             IndexedEvent e = eventsSrc.lines[i].indexedEvent[eventIndex];
//             BBox bbox = eventsSrc.lines[i].ranges[eventIndex];
//             CTuint primId = eventsSrc.lines[i].primId[eventIndex];
//             CTeventType_t type = eventsSrc.lines[i].type[eventIndex];
// 
//             CTaxis_t splitAxis = getAxisFromMask(masks[i]);
//             bool right = isRight(masks[i]);
// 
//             CTuint nnodeIndex;
// 
//             if(i == 0)
//             {
//                 nnodeIndex = eventsSrc.lines[i].nodeIndex[eventIndex];
//             }
// 
//             if(isOLappin(masks[i]))
//             {
//                 CTreal split = cms[i].newSplit[id];
//                 setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
//                 if(i == splitAxis && ((masks[i] & 0x40) == 0x40))
//                 {
//                     e.v = split;
//                 }
//             }
// 
//             eventsDst.lines[i].indexedEvent[dstAdd] = e;
//             eventsDst.lines[i].primId[dstAdd] = primId;
//             eventsDst.lines[i].ranges[dstAdd] = bbox;
//             eventsDst.lines[i].type[dstAdd] = type;
// 
//             if(i == 0)
//             {
//                 eventsDst.lines[i].nodeIndex[dstAdd] = 2 * nnodeIndex + (CTuint)right;
//             }
//         }
//     }
// }
// 
// struct CompactionTestData
// {
// 
// };
// 
// 


#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <iostream>
#include <iterator>
#include <string>

struct DDate
{
//     IndexedEvent e;
//     BBox bbox;
//     uint primId;
//     byte type;
    uint nnodeIndex;
    uint mask;
};

// this functor returns true if the argument is odd, and false otherwise
template <typename T>
struct is_odd : public thrust::unary_function<T,bool>
{
    __host__ __device__
    bool operator()(T x)
    {
        return x.mask == 1;
    }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));  
    std::cout << "\n";
}

__host__ static __inline__ DDate genData()
{
    DDate d;
    d.mask = (rand()/(float)RAND_MAX) < 0.01 ? 1 : 0;
    return d;
}

int main(void)
{
    // input size
    size_t N = 298304;

    thrust::host_vector<DDate> h_1(N);
    srand (time(NULL));
    thrust::generate(h_1.begin(), h_1.end(), genData);


    // define some types
    typedef thrust::device_vector<DDate> Vector;
    typedef Vector::iterator           Iterator;

    // allocate storage for array
    Vector values(N);

    // initialize array to [0, 1, 2, ... ]
    //thrust::sequence(values.begin(), values.end());
    values = h_1;
    
    //print_range("values", values.begin(), values.end());

    // allocate output storage, here we conservatively assume all values will be copied
    Vector output(values.size());
    chimera::util::HTimer timer;
    cudaDeviceSynchronize();
    timer.Start();
    // copy odd numbers to separate array
    for(int i = 0; i < 22; ++i)
    {
        thrust::copy_if(values.begin(), values.end(), output.begin(), is_odd<DDate>());
    }
    cudaDeviceSynchronize();
    timer.Stop();
    printf("%f \n", timer.GetMillis());
    system("pause");

    //print_range("output", output.begin(), output_end);

    // another approach is to count the number of values that will 
    // be copied, and allocate an array of the right size
//     size_t N_odd = thrust::count_if(values.begin(), values.end(), is_odd<int>());
//     
//     Vector small_output(N_odd);
//     
//     thrust::copy_if(values.begin(), values.end(), small_output.begin(), is_odd<int>());
//     
//     //print_range("small_output", small_output.begin(), small_output.end());
// 
//     // we can also compact sequences with the remove functions, which do the opposite of copy
//     thrust::remove_if(values.begin(), values.end(), is_odd<int>());

    // since the values after values_end are garbage, we'll resize the vector
    //values.resize(values_end - values.begin());

    //print_range("values", values.begin(), values.end());

    return 0;
}