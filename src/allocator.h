#pragma once

#include "logger.h"
#include <stddef.h>
#include <stdint.h>
#include <iostream>

// Overloaded operators.
// https://blog.molecular-matters.com/2011/07/07/memory-system-part-2/

template <typename ALLOCATOR>
void* operator new(size_t size, ALLOCATOR* a, int line, const char* file)
{
#if defined(NIMBLE_TRACK_ALLOCATIONS)
    NIMBLE_LOG_INFO("Performing allocation of size " + size + " at line " + line + " of " + file);
#endif
	return a->allocate(size, 8);
}

template <typename OBJECT, typename ALLOCATOR>
OBJECT* custom_new_array(size_t size, ALLOCATOR* a, int line, const char* file)
{
#if defined(NIMBLE_TRACK_ALLOCATIONS)
    NIMBLE_LOG_INFO("Performing array allocation of size " + size + " at line " + line + " of " + file);
#endif
	union
	{
		void*	as_void;
		size_t* as_size_t;
		OBJECT* as_T;
	};

	as_void = a->allocate(sizeof(OBJECT) * size + sizeof(size_t), 8);

	// store number of instances in first size_t bytes
	*as_size_t++ = size;

	// construct instances using placement new
	const OBJECT* const one_past_last = as_T + size;
	while (as_T < one_past_last)
		new (as_T++) OBJECT;

	// hand user the pointer to the first instance
	return (as_T - size);
}

template <typename OBJECT, typename ALLOCATOR>
void custom_delete(OBJECT* p, ALLOCATOR* a, int line, const char* file)
{
#if defined(NIMBLE_TRACK_ALLOCATIONS)
    NIMBLE_LOG_INFO("Performing deallocation of size " + sizeof(OBJECT) + " at line " + line + " of " + file);
#endif
	p->~OBJECT();
	a->deallocate(p);
}

template <typename OBJECT, typename ALLOCATOR>
void custom_delete_array(OBJECT* p, ALLOCATOR* a, int line, const char* file)
{
	union
	{
		size_t* as_size_t;
		OBJECT* as_T;
	};

	// user pointer points to first instance...
	as_T = p;

	// ...so go back size_t bytes and grab number of instances
	const size_t N = as_size_t[-1];

#if defined(NIMBLE_TRACK_ALLOCATIONS)
	NIMBLE_LOG_INFO("Performing array deallocation of size " + (N * sizeof(OBJECT)) + " at line " + line + " of " + file);
#endif

	// call instances' destructor in reverse order
	for (size_t i = N; i>0; --i)
		as_T[i - 1].~OBJECT();

	a->deallocate(as_size_t - 1);
}

#define NIMBLE_NEW(ALLOCATOR) new(ALLOCATOR, __LINE__, __FILE__)
#define NIMBLE_DELETE(OBJECT, ALLOCATOR) custom_delete(OBJECT, ALLOCATOR, __LINE__, __FILE__)
		
#define NIMBLE_NEW_ARRAY(CLASS, N, ALLOCATOR) custom_new_array<CLASS>(N, ALLOCATOR, __LINE__, __FILE__)
#define NIMBLE_DELETE_ARRAY(OBJECT, ALLOCATOR) custom_delete_array(OBJECT, ALLOCATOR, __LINE__, __FILE__)

namespace nimble
{
namespace memory
{
inline void* align_forward(void* address, uint8_t alignment)
{
    return (void*)((reinterpret_cast<uintptr_t>(address) + static_cast<uintptr_t>(alignment - 1)) & (static_cast<uintptr_t>(~(alignment - 1))));
}

inline const void* align_forward(const void* address, uint8_t alignment)
{
    return (void*)((reinterpret_cast<uintptr_t>(address) + static_cast<uintptr_t>(alignment - 1)) & (static_cast<uintptr_t>(~(alignment - 1))));
}

inline void* align_backward(void* address, uint8_t alignment)
{
    return (void*)(reinterpret_cast<uintptr_t>(address) & static_cast<uintptr_t>(~(alignment - 1)));
}

inline const void* align_backward(const void* address, uint8_t alignment)
{
    return (void*)(reinterpret_cast<uintptr_t>(address) & static_cast<uintptr_t>(~(alignment - 1)));
}

inline uint8_t align_foward_adjustment(void* address, uint8_t alignment)
{
    uint8_t adjustment = alignment - (reinterpret_cast<uintptr_t>(address) & static_cast<uintptr_t>(alignment - 1));

    if (adjustment == alignment)
        return 0;

    return adjustment;
}

inline uint8_t align_foward_adjustment_with_header(void* address, uint8_t alignment, uint8_t header_size)
{
    uint8_t adjustment  = align_foward_adjustment(address, alignment);
    uint8_t neededSpace = header_size;

    if (adjustment < neededSpace)
    {
        neededSpace -= adjustment;
        adjustment += alignment * (neededSpace / alignment);

        if (neededSpace % alignment > 0)
            adjustment += alignment;
    }

    return adjustment;
}

inline uint8_t align_backward_adjustment(void* address, uint8_t alignment)
{
    uint8_t adjustment = (reinterpret_cast<uintptr_t>(address) & static_cast<uintptr_t>(~(alignment - 1)));

    if (adjustment == alignment)
        return 0;

    return adjustment;
}

inline void* add(void* pointer, size_t size)
{
    return (void*)(reinterpret_cast<uintptr_t>(pointer) + size);
}

inline const void* add(const void* pointer, size_t size)
{
    return (const void*)(reinterpret_cast<uintptr_t>(pointer) + size);
}

inline void* subtract(void* pointer, size_t size)
{
    return (void*)(reinterpret_cast<uintptr_t>(pointer) - size);
}

inline const void* subtract(const void* pointer, size_t size)
{
    return (const void*)(reinterpret_cast<uintptr_t>(pointer) - size);
}
} // namespace memory

class Allocator
{
public:
	Allocator() : m_size(0), m_memory(nullptr), m_used_size(0), m_num_allocations(0) {}
	virtual ~Allocator() {}
	virtual void* allocate(size_t size, size_t align) = 0;
	virtual void deallocate(void* ptr) = 0;

	inline size_t num_allocations() { return m_num_allocations; }
	inline size_t allocated_size() { return m_used_size; }
	
protected:
	size_t m_size;
	size_t m_used_size;
	size_t m_num_allocations;
	void*  m_memory;
};

} // namespace nimble
