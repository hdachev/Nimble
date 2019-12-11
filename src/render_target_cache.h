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

    uint32_t                   id;
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
    struct TemporaryRenderTargetBank
    {
        uint32_t                            allocated_count;
        uint32_t                            created_count;
        std::vector<TemporaryRenderTarget*> rts;

        TemporaryRenderTargetBank();
        ~TemporaryRenderTargetBank();
        TemporaryRenderTarget* request_temporary(uint32_t w, uint32_t h, GLenum format);
        void                   release_temporary(TemporaryRenderTarget* rt);
        bool                   exists(uint32_t id);
    };

    std::unordered_map<uint64_t, TemporaryRenderTargetBank> m_cache;
};
} // namespace nimble