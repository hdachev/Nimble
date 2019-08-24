#include "render_target_cache.h"

#define TEMP_RT_CACHE_SIZE 100

namespace nimble
{
static uint32_t g_rt_idx = 0;

RenderTargetCache::RenderTargetCache()
{
}

RenderTargetCache::~RenderTargetCache()
{
}

TemporaryRenderTarget* RenderTargetCache::request_temporary(uint32_t w, uint32_t h, GLenum format)
{
}

void RenderTargetCache::release_temporary(TemporaryRenderTarget* rt)
{
}

void RenderTargetCache::clear()
{
    m_cache.clear();
}

RenderTargetCache::TemporaryRenderTargetBank::TemporaryRenderTargetBank()
{
    allocated_count = 0;
    created_count   = 0;
    rts.resize(TEMP_RT_CACHE_SIZE);
}

RenderTargetCache::TemporaryRenderTargetBank::~TemporaryRenderTargetBank()
{
    for (auto rt : rts)
    {
        delete rt;
        rt = nullptr;
	}

	rts.clear();
}

TemporaryRenderTarget* RenderTargetCache::TemporaryRenderTargetBank::request_temporary(uint32_t w, uint32_t h, GLenum format)
{
    if (created_count == allocated_count)
    {
        TemporaryRenderTarget* rt = new TemporaryRenderTarget();

        rt->id          = g_rt_idx++;
        rt->desc.w      = w;
        rt->desc.h      = h;
        rt->desc.format = format;

        rts.push_back(rt);
        created_count++;
    }

    return rts[allocated_count++];
}

void RenderTargetCache::TemporaryRenderTargetBank::release_temporary(TemporaryRenderTarget*& rt)
{
    // Make sure the provided render target has not been already released
    if (allocated_count == 0 || exists(rt->id))
        return;

    rts[--allocated_count] = rt;
    rt                     = nullptr;
}

bool RenderTargetCache::TemporaryRenderTargetBank::exists(uint32_t id)
{
    for (uint32_t i = (allocated_count - 1); i < rts.size(); i++)
    {
        if (rts[i]->id == id)
            return true;
    }

    return false;
}
} // namespace nimble