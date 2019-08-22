#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "murmur_hash.h"
#include "ogl.h"

namespace nimble
{
struct TemporaryRenderTarget
{
    struct Desc
    {
        uint32_t w;
        uint32_t h;
        GLenum   format;
    };

    std::shared_ptr<Texture2D> texture;
    Desc                       desc;
};

class RenderTargetCache
{
public:
    RenderTargetCache();
    ~RenderTargetCache();

    TemporaryRenderTarget* request_temporary(uint32_t w, uint32_t h, GLenum format);
    void                   release_temporary(TemporaryRenderTarget* rt);
    void                   clear();

private:
    std::unordered_map<uint64_t, std::vector<TemporaryRenderTarget>> m_cache;
};
} // namespace nimble