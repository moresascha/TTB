#include <thrust/sort.h>
#include <thrust/detail/type_traits.h>
#include "cuKDTree.h"

void SortEvents(EventLines* eventLine)
{
    for(CTbyte i = 0; i < 3; ++i)
    {
        thrust::sort_by_key(eventLine->eventLines[i].rawEvents->begin(), eventLine->eventLines[i].rawEvents->end(),  eventLine->eventLines[i].eventKeys->begin());
    }
}