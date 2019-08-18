#pragma once

#include "ogl.h"
#include "mesh.h"
#include "static_hash_map.h"
#include "shader_key.h"

namespace nimble
{
#define MAX_SHADER_CACHE_SIZE 10000
#define MAX_PROGRAM_CACHE_SIZE 10000

class ShaderLibrary
{
public:
    ShaderLibrary(std::vector<std::pair<GLenum, std::string>> shaders);
    ~ShaderLibrary();

    Program* create_program(const std::vector<std::string>& defines);

private:
    StaticHashMap<uint64_t, Program*, MAX_PROGRAM_CACHE_SIZE> m_program_cache;
    std::vector<std::pair<GLenum, std::string>>               m_sources;
};
} // namespace nimble