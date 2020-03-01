#pragma once

#include "allocator.h"

namespace nimble
{
class LinearAllocator : public Allocator
{
public:
    LinearAllocator(void* memory, size_t max_size);
    virtual ~LinearAllocator();
    void* allocate(size_t size, size_t align) override;
    void  deallocate(void* ptr) override;
    void  clear();

protected:
    void* m_position;
};
} // namespace nimble
