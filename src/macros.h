#pragma once

#if defined(_MSC_VER)
#define NIMBLE_ALIGNED(x) __declspec(align(x))
#else
#if defined(__GNUC__) || defined(__clang__)
#define NIMBLE_ALIGNED(x) __attribute__ ((aligned(x)))
#endif
#endif

#define NIMBLE_ZERO_MEMORY(x) memset(&x, 0, sizeof(x))

#define NIMBLE_SAFE_DELETE(x) if(x) { delete x; x = nullptr; }
#define NIMBLE_SAFE_DELETE_ARRAY(x) if(x) { delete[] x; x = nullptr; }