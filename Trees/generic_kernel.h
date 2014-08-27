
#ifdef FAST_COMPACTION
#define MAKE_FN_NAME(x) __global__ void  optimizedcompactEventLineV3Type ## x (CTbyte srcIndex, CTuint tilesPerLanes, CTuint activeLanes, CTuint N)
#else
#define MAKE_FN_NAME(x) __global__ void  optimizedcompactEventLineV3Type ## x (CTbyte srcIndex, CTuint N)
#endif

#define FUNCTION_NAME(__name) MAKE_FN_NAME(__name)

#ifdef FAST_COMPACTION

template<int blockSize> 
    FUNCTION_NAME(FUNCNAME)
{

#if PATH == 0
    __shared__ IndexedEvent s_indexEvents[blockSize];
#elif PATH == 1
    __shared__ BBox s_bboxes[blockSize];
#elif PATH == 2
    __shared__ CTuint s_primIds[blockSize];
#elif PATH == 3
    __shared__ CTeventType_t s_eventTypes[blockSize];
#elif PATH == 4
    __shared__ CTuint s_nodeIndices[blockSize];
#endif

    __shared__ CTuint s_scanned[blockSize];

    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    int laneId = threadIdx.x;
    int warpId = blockIdx.x;

    if(warpId >= activeLanes)
    {
        return;
    }

#if PATH == 0
    IndexedEvent e;
#elif PATH == 1
    BBox bbox;
#elif PATH == 2
    CTuint primId;
#elif PATH == 3
    CTeventType_t type;
#elif PATH == 4
    CTuint nnodeIndex;
#endif

    CTuint blockPrefix;

#if PATH != 4
#pragma unroll
    for(int axis = 0; axis < 3; ++axis)
    {
#else
    int axis = 0;
#endif
    CTuint buffered = 0;

    if(cms[axis].elemsPerTile[warpId] == 0)
    {
#if PATH != 4
        continue;
#else
        return;
#endif
    }

    CTuint tileOffset = cms[axis].scanned[warpId];
    int elemCount;

    for(int i = 0; i < tilesPerLanes; ++i)
    {
        int addr = blockSize * tilesPerLanes * warpId + blockSize * i + laneId;
        CTclipMask_t mask = (addr >= N ? 0 : cms[axis].mask[addr]);
        bool right = isRight(mask);

        blockPrefix = __blockBinaryPrefixSums(s_scanned, mask > 0);

        elemCount = s_scanned[blockSize/32];

        if(mask)
        {
            CTuint eventIndex = cms[axis].index[addr];
#if PATH == 0
            e = eventsSrc.lines[axis].indexedEvent[eventIndex];
#elif PATH == 1
            bbox = eventsSrc.lines[axis].ranges[eventIndex];
#elif PATH == 2
            primId = eventsSrc.lines[axis].primId[eventIndex];
#elif PATH == 3
            type = eventsSrc.lines[axis].type[eventIndex];
#elif PATH == 4
            nnodeIndex = 2 * eventsSrc.lines[axis].nodeIndex[eventIndex] + (CTuint)right;
#endif

#if PATH == 0
            CTaxis_t splitAxis = getAxisFromMask(mask);

            if(isOLappin(mask))
            {
                BBox bbox = eventsSrc.lines[axis].ranges[eventIndex];
                CTreal split = cms[axis].newSplit[addr];
                setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
                if(i == splitAxis && ((mask & 0x40) == 0x40))
                {
                    e.v = split;
                }
            }
#endif

            if(blockPrefix + buffered < blockSize)
            {

#if PATH == 0
                *(s_indexEvents + blockPrefix + buffered) = e;
#elif PATH == 1
                *(s_bboxes      + blockPrefix + buffered) = bbox;
#elif PATH == 2
                *(s_primIds     + blockPrefix + buffered) = primId;
#elif PATH == 3
                *(s_eventTypes  + blockPrefix + buffered) = type;
#elif PATH == 4
                *(s_nodeIndices + blockPrefix + buffered) = nnodeIndex;
#endif               
            }
        }

        __syncthreads();

        if(buffered + elemCount >= blockSize)
        {
#if PATH == 0
            eventsDst.lines[axis].indexedEvent[tileOffset + laneId] = s_indexEvents[laneId];
#elif PATH == 1
            eventsDst.lines[axis].ranges      [tileOffset + laneId] = s_bboxes     [laneId];
#elif PATH == 2
            eventsDst.lines[axis].primId      [tileOffset + laneId] = s_primIds    [laneId];
#elif PATH == 3
            eventsDst.lines[axis].type        [tileOffset + laneId] = s_eventTypes [laneId];
#elif PATH == 4
            eventsDst.lines[axis].nodeIndex[tileOffset + laneId] = s_nodeIndices[laneId];
#endif
            tileOffset += blockSize;
        }

        __syncthreads();

        if(buffered + blockPrefix >= blockSize &&  mask)
        {
#if PATH == 0
            *(s_indexEvents + blockPrefix + buffered - blockSize) = e;
#elif PATH == 1
            *(s_bboxes      + blockPrefix + buffered - blockSize) = bbox;
#elif PATH == 2
            *(s_primIds     + blockPrefix + buffered - blockSize) = primId;
#elif PATH == 3
            *(s_eventTypes  + blockPrefix + buffered - blockSize) = type;
#elif PATH == 4
            *(s_nodeIndices +  blockPrefix + buffered - blockSize) = nnodeIndex;
#endif
        }

        buffered = (buffered + elemCount) % blockSize;
    }

    __syncthreads();

    if(laneId < buffered)
    {
#if PATH == 0
        eventsDst.lines[axis].indexedEvent[tileOffset + laneId] = s_indexEvents[laneId];
#elif PATH == 1
        eventsDst.lines[axis].ranges      [tileOffset + laneId] = s_bboxes     [laneId];
#elif PATH == 2
        eventsDst.lines[axis].primId      [tileOffset + laneId] = s_primIds    [laneId];
#elif PATH == 3
        eventsDst.lines[axis].type        [tileOffset + laneId] = s_eventTypes [laneId];
#elif PATH == 4
        eventsDst.lines[axis].nodeIndex[tileOffset + laneId] = s_nodeIndices[laneId];
#endif
    }
#if PATH != 4
}
#endif
}

#else

FUNCTION_NAME(FUNCNAME)
{
    RETURN_IF_OOB(N);
    EVENT_TRIPLE_HEADER_SRC;
    EVENT_TRIPLE_HEADER_DST;

    CTuint masks[3];
    masks[0] = cms[0].mask[id];
    masks[1] = cms[1].mask[id];
    masks[2] = cms[2].mask[id];

    if(masks[0] == 0 && masks[1] == 0 && masks[2] == 0)
    {
        return;
    }

#pragma unroll
    for(CTaxis_t axis = 0; axis < 3; ++axis)
    {
        if(isSet(masks[axis]))
        {
            CTuint eventIndex = cms[axis].index[id];

            CTuint dstAdd = cms[axis].scanned[id];

#if PATH == 0
            IndexedEvent e = eventsSrc.lines[axis].indexedEvent[eventIndex];
#elif PATH == 1
            BBox bbox = eventsSrc.lines[axis].ranges[eventIndex];
#elif PATH == 2
            CTuint primId = eventsSrc.lines[axis].primId[eventIndex];
#elif PATH == 3
            CTeventType_t type = eventsSrc.lines[axis].type[eventIndex];
#endif
            bool right = isRight(masks[axis]);
#if PATH == 4
            CTuint nnodeIndex;
            if(axis == 0)
            {
                nnodeIndex = 2 * eventsSrc.lines[axis].nodeIndex[eventIndex] + (CTuint)right;
            }
#endif

#if PATH == 0
            CTaxis_t splitAxis = getAxisFromMask(masks[axis]);
            if(isOLappin(masks[axis]))
            {
                BBox bbox = eventsSrc.lines[axis].ranges[eventIndex];
                CTreal split = cms[axis].newSplit[id];
                setAxis(right ? bbox.m_min : bbox.m_max, splitAxis, split);
                if(axis == splitAxis && ((masks[axis] & 0x40) == 0x40))
                {
                    e.v = split;
                }
            }
#endif

#if PATH == 0
            eventsDst.lines[axis].indexedEvent[dstAdd] = e;
#elif PATH == 1
            eventsDst.lines[axis].ranges[dstAdd] = bbox;
#elif PATH == 2
            eventsDst.lines[axis].primId[dstAdd] = primId;
#elif PATH == 3
            eventsDst.lines[axis].type[dstAdd] = type;
#elif PATH == 4
            if(axis == 0)
            {
                eventsDst.lines[axis].nodeIndex[dstAdd] = nnodeIndex;
            }
#endif
        }
    }
}
#endif