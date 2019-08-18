#include "shader_library.h"
#include "utility.h"
#include "logger.h"
#include "render_node.h"
#include "renderer.h"
#include "render_graph.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

ShaderLibrary::ShaderLibrary(std::vector<std::pair<GLenum, std::string>> shaders)
{
	for (auto& pair : shaders)
	{
		std::string source;

		if (!utility::read_shader(utility::path_for_resource("assets/" + pair.second), source))
			NIMBLE_LOG_ERROR("Failed load Shader Library Source: " + pair.second);

		m_sources.push_back({ pair.first, source });
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

ShaderLibrary::~ShaderLibrary()
{
    for (uint32_t i = 0; i < m_program_cache.size(); i++)
    {
        NIMBLE_SAFE_DELETE(m_program_cache.m_value[i]);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

Program* ShaderLibrary::create_program(const std::vector<std::string>& defines)
{
    std::vector<Shader*> shaders;

	std::string defines_string = "";

	for (auto str : defines)
	{
		defines_string += str;
		defines_string += "\n";
	}

	for (auto& pair : m_sources)
	{
		std::string source = defines_string + pair.second;
	}
	
	std::string fs_source = defines_string + m_fs_source;

	vs = new Shader(GL_VERTEX_SHADER, vs_source);
	fs = new Shader(GL_FRAGMENT_SHADER, fs_source);

    Shader* shaders[] = { vs, fs };

    if (vs->compiled() && fs->compiled())
    {
        Program* program = new Program(2, shaders);

        program->uniform_block_binding("u_PerEntity", 1);

		uint64_t hash = NIMBLE_HASH(defines_string.c_str());
        m_program_cache.set(hash, program);

        return program;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble