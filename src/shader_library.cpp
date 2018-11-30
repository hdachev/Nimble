#include "shader_library.h"
#include "utility.h"
#include "logger.h"

namespace nimble
{
	const std::string kMeshTypeLUT[] =
	{
		"#define MESH_TYPE_STATIC",
		"#define MESH_TYPE_SKELETAL"
	};

	const std::string kDisplacementTypeLUT[] =
	{
		"#define DISPLACEMENT_TYPE_PARALLAX_OCCLUSION",
		"#define DISPLACEMENT_TYPE_TESSELLATION"
	};

	static const std::string kSurfaceTextureLUT[] =
	{
		"#define TEXTURE_ALBEDO",
		"#define TEXTURE_NORMAL",
		"#define TEXTURE_METALLIC",
		"#define TEXTURE_ROUGHNESS",
		"#define TEXTURE_SPECULAR",
		"#define TEXTURE_SMOOTHNESS",
		"#define TEXTURE_DISPLACEMENT"
	};

	static const std::string kCustomTextureLUT[] =
	{
		"#define TEXTURE_CUSTOM_1",
		"#define TEXTURE_CUSTOM_2",
		"#define TEXTURE_CUSTOM_3",
		"#define TEXTURE_CUSTOM_4",
		"#define TEXTURE_CUSTOM_5",
		"#define TEXTURE_CUSTOM_6",
		"#define TEXTURE_CUSTOM_7",
		"#define TEXTURE_CUSTOM_8"
	};

	// -----------------------------------------------------------------------------------------------------------------------------------

	ShaderLibrary::ShaderLibrary(const std::string& vs, const std::string& fs)
	{
		if (!utility::read_shader(utility::path_for_resource("assets/" + vs), m_vs_template))
			NIMBLE_LOG_ERROR("Failed load Shader Library VS Source: " + vs);

		if (!utility::read_shader(utility::path_for_resource("assets/" + fs), m_fs_template))
			NIMBLE_LOG_ERROR("Failed load Shader Library FS Source: " + fs);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ShaderLibrary::~ShaderLibrary()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Program* ShaderLibrary::lookup_program(const ProgramKey& key)
	{
		Program* program = nullptr;

		if (m_program_cache.has(key.key))
			m_program_cache.get(key.key, program);
		
		return program;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Program* ShaderLibrary::create_program(const MeshType& type, const std::shared_ptr<Material>& material)
	{
		std::string vs_template = m_vs_template;
		std::string fs_template = m_fs_template;

		std::vector<std::string> vs_defines;
		std::vector<std::string> fs_defines;

		ProgramKey program_key = material->program_key();
		VertexShaderKey vs_key = material->vs_key();
		FragmentShaderKey fs_key = material->fs_key();

		program_key.set_mesh_type(type);
		vs_key.set_mesh_type(type);

		Shader* vs = nullptr;
		Shader* fs = nullptr;

		// COMMON

		// Normal Texture
		if (material->surface_texture(TEXTURE_TYPE_NORMAL))
		{
			vs_defines.push_back(kSurfaceTextureLUT[TEXTURE_TYPE_NORMAL]);
			fs_defines.push_back(kSurfaceTextureLUT[TEXTURE_TYPE_NORMAL]);
		}

		// VERTEX SHADER
		if (m_vs_cache.find(vs_key.key) != m_vs_cache.end())
			vs = m_vs_cache[vs_key.key];
		else
		{
			std::string source;

			// Mesh Type
			vs_defines.push_back(kMeshTypeLUT[type]);

			// Vertex Func
			if (material->has_vertex_shader_func())
			{
				source = "#ifndef VERTEX_SHADER_FUNC\n";
				source += "#define VERTEX_SHADER_FUNC\n";
				source += material->vertex_shader_func();
				source += "#endif\n\n";
				source += vs_template;
			}

			std::string defines_str = "";

			for (auto& define : vs_defines)
			{
				defines_str += define;
				defines_str += "\n";
			}

			source = defines_str + source;

			vs = new Shader(GL_VERTEX_SHADER, source.c_str());

			m_vs_cache[vs_key.key] = vs;
		}

		// FRAGMENT SHADER
		if (m_fs_cache.find(fs_key.key) != m_fs_cache.end())
			fs = m_vs_cache[fs_key.key];
		else
		{
			std::string source;

			// Displacement
			if (material->surface_texture(TEXTURE_TYPE_DISPLACEMENT))
				fs_defines.push_back(kDisplacementTypeLUT[type]);

			// Custom Textures
			for (uint32_t i = 0; i < material->custom_texture_count(); i++)
				fs_defines.push_back(kCustomTextureLUT[i]);

			// Fragment Func
			if (material->has_fragment_shader_func())
			{
				source = "#ifndef FRAGMENT_SHADER_FUNC\n";
				source += "#define FRAGMENT_SHADER_FUNC\n";
				source += material->fragment_shader_func();
				source += "#endif\n\n";
				source += fs_template;
			}

			std::string defines_str = "";

			for (auto& define : fs_defines)
			{
				defines_str += define;
				defines_str += "\n";
			}

			source = defines_str + source;

			fs = new Shader(GL_FRAGMENT_SHADER, source.c_str());

			m_fs_cache[fs_key.key] = fs;
		}

		Shader* shaders[] = { vs, fs };
		Program* program = new Program(2, shaders);

		m_program_cache.set(program_key.key, program);

		return program;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble