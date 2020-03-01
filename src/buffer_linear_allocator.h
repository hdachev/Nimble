#pragma once

#include <stdint.h>
#include <assert.h>
#include "linear_allocator.h"

namespace nimble
{
template <size_t SIZE>
class BufferLinearAllocator : public LinearAllocator
{
public:
    BufferLinearAllocator() :
        LinearAllocator(&m_buffer[0], SIZE) {}

private:
    uint8_t m_buffer[SIZE];
};
} // namespace nimble
