#include "render_target_cache.h"

namespace nimble
{
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
} // namespace nimble