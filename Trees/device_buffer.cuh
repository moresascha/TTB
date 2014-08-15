#pragma once

extern "C" __device__ CTuint mem_alloc_error;

template <typename T>
struct cuHeapBuffer
{
    T* _pMemPtr;
    size_t _size;

    __device__ __host__ cuHeapBuffer(void)
    {

    }

    __device__ __host__ void init(void)
    {
        if(threadIdx.x > 0) return;
        _pMemPtr = NULL;
        _size = 0;
    }

    __device__ __host__ void resize(size_t newSize)
    {
        if(threadIdx.x > 0) return;

        if(_size < newSize)
        {
            T* newPtr; 
            cudaError_t valid = cudaMalloc(&newPtr, sizeof(T) * newSize);
            
//             if(valid != cudaSuccess)
//             {
//                 mem_alloc_error = (int)valid;
//             }

            if(_pMemPtr)
            {
                memcpy((void*)newPtr, (void*)_pMemPtr, sizeof(T) * _size);
                cudaFree(_pMemPtr);
            }

            _pMemPtr = newPtr;
            _size = newSize;
        }
    }

    __device__ __host__ T& operator[](size_t index)
    {
        return _pMemPtr[index];
    }

    __device__ __host__ size_t size(void)
    {
        return _size;
    }

    __device__ __host__ void destroy(void)
    {
        if(threadIdx.x > 0) return;

        if(_pMemPtr)
        {
            cudaFree(_pMemPtr);
            _pMemPtr = NULL;
            _size = 0;
        }
    }

    __device__ __host__ T* GetPointer(void)
    {
        return _pMemPtr;
    }

    __device__ __host__ const T* __restrict GetConstPointer(void)
    {
        return _pMemPtr;
    }

    __device__ __host__ ~cuHeapBuffer(void)
    {

    }
};

template <typename T>
struct cuHeapDoubleBuffer
{
     CTbyte _current;
     cuHeapBuffer<T> _buffer0;
     cuHeapBuffer<T> _buffer1;

    __device__ __host__ cuHeapDoubleBuffer(void)
    {

    }

    __device__ __host__ ~cuHeapDoubleBuffer(void)
    {

    }

    __device__ __host__ void init(void)
    {
        _buffer0.init();
        _buffer1.init();
        _current = 0;
    }

    __device__ __host__ void resize(size_t size)
    {
        _buffer0.resize(size);
        _buffer1.resize(size);
    }

    __device__ __host__ cuHeapBuffer<T>& operator[](CTbyte index)
    {
        return index == 0 ? _buffer0 : _buffer1;
    }

    __device__ __host__ size_t size(void)
    {
        return _buffer0.size();
    }
};