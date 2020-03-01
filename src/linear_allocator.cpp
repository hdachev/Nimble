#include "linear_allocator.h"
#include <assert.h>

namespace nimble
{
LinearAllocator::LinearAllocator(void* memory, size_t max_size) : Allocator()
{
	// Only init once.
	assert(m_memory == nullptr);
	assert(memory);
	assert(max_size > 0);

	m_size = max_size;
	m_memory = memory;
	m_position = m_memory;
}

LinearAllocator::~LinearAllocator()
{

}

void* LinearAllocator::allocate(size_t size, size_t align)
{
	assert(size != 0);

	uint8_t adjustment = memory::align_backward_adjustment(m_position, align);

	if (m_used_size + adjustment + size > m_size)
		return nullptr;

	uintptr_t aligned_address = (uintptr_t)m_position + adjustment;
	m_position = (void*)(aligned_address + size);
	m_num_allocations++;
	m_used_size += adjustment + size;

	return (void*)aligned_address;
}

void LinearAllocator::deallocate(void* ptr)
{
	assert(false);
}

void LinearAllocator::clear()
{
	m_num_allocations = 0;
	m_used_size		  = 0;
	m_position		  = m_memory;
}

} // namespace nimble
