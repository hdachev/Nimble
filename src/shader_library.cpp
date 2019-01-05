#include "shader_library.h"
#include "utility.h"
#include "logger.h"
#include "render_node.h"

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
		"#define TEXTURE_METAL_SPEC",
		"#define TEXTURE_ROUGH_SMOOTH",
		"#define TEXTURE_DISPLACEMENT",
		"#define TEXTURE_EMISSIVE"
	};

	static const std::string kMaterialUniformLUT[] =
	{
		"#define UNIFORM_ALBEDO",
		"",
		"#define UNIFORM_METAL_SPEC",
		"#define UNIFORM_ROUGH_SMOOTH",
		"",
		"#define UNIFORM_EMISSIVE"
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

	static const RenderNodeFlags kRenderNodeFlags[] =
	{
		NODE_USAGE_MATERIAL_ALBEDO,
		NODE_USAGE_MATERIAL_NORMAL,
		NODE_USAGE_MATERIAL_METAL_SPEC,
		NODE_USAGE_MATERIAL_ROUGH_SMOOTH,
		NODE_USAGE_MATERIAL_DISPLACEMENT,
		NODE_USAGE_MATERIAL_EMISSIVE
	};

	// -----------------------------------------------------------------------------------------------------------------------------------

	ShaderLibrary::ShaderLibrary(const std::string& vs, const std::string& fs)
	{
		if (!utility::read_shader_separate(utility::path_for_resource("assets/" + vs), m_vs_template_includes, m_vs_template_source, m_vs_template_defines))
			NIMBLE_LOG_ERROR("Failed load Shader Library VS Source: " + vs);

		if (!utility::read_shader_separate(utility::path_for_resource("assets/" + fs), m_fs_template_includes, m_fs_template_source, m_fs_template_defines))
			NIMBLE_LOG_ERROR("Failed load Shader Library FS Source: " + fs);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ShaderLibrary::~ShaderLibrary()
	{
		for (uint32_t i = 0; i < m_program_cache.size(); i++)
		{
			NIMBLE_SAFE_DELETE(m_program_cache.m_value[i]);
		}

		for (auto pair : m_vs_cache)
		{
			NIMBLE_SAFE_DELETE(pair.second);
		}

		for (auto pair : m_fs_cache)
		{
			NIMBLE_SAFE_DELETE(pair.second);
		}
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

	Program* ShaderLibrary::create_program(const MeshType& type, const uint32_t& flags, const std::shared_ptr<Material>& material)
	{
		std::string vs_template = m_vs_template_source;
		std::string fs_template = m_fs_template_source;

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

		if (HAS_BIT_FLAG(flags, NODE_USAGE_PER_OBJECT_UBO))
		{
			vs_defines.push_back("#define PER_OBJECT_UBO");
			fs_defines.push_back("#define PER_OBJECT_UBO");
		}
		if (HAS_BIT_FLAG(flags, NODE_USAGE_PER_VIEW_UBO))
		{
			vs_defines.push_back("#define PER_VIEW_UBO");
			fs_defines.push_back("#define PER_VIEW_UBO");
		}
		if (HAS_BIT_FLAG(flags, NODE_USAGE_PER_SCENE_UBO))
		{
			vs_defines.push_back("#define PER_SCENE_UBO");
			fs_defines.push_back("#define PER_SCENE_UBO");
		}

		// Normal Texture
		if (material->surface_texture(TEXTURE_TYPE_NORMAL) && HAS_BIT_FLAG(flags, NODE_USAGE_MATERIAL_NORMAL))
		{
			vs_defines.push_back(kSurfaceTextureLUT[TEXTURE_TYPE_NORMAL]);
			fs_defines.push_back(kSurfaceTextureLUT[TEXTURE_TYPE_NORMAL]);
		}

		// Displacement
		if (material->surface_texture(TEXTURE_TYPE_DISPLACEMENT) && HAS_BIT_FLAG(flags, NODE_USAGE_MATERIAL_DISPLACEMENT))
		{
			vs_defines.push_back(kDisplacementTypeLUT[type]);
			fs_defines.push_back(kDisplacementTypeLUT[type]);
		}

		// VERTEX SHADER
		if (m_vs_cache.find(vs_key.key) != m_vs_cache.end())
			vs = m_vs_cache[vs_key.key];
		else
		{
			std::string source;

			// Mesh Type
			vs_defines.push_back(kMeshTypeLUT[type]);

			// Vertex Source
			source = m_vs_template_defines;
			source += "\n\n";
			source += m_vs_template_includes;
			source += "\n\n";

			if (material->has_vertex_shader_func())
			{
				source += "#ifndef VERTEX_SHADER_FUNC\n";
				source += "#define VERTEX_SHADER_FUNC\n";
				source += material->vertex_shader_func();
				source += "#endif\n\n";
			}
			
			source += vs_template;

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

			if (material->blend_mode() == BLEND_MODE_MASKED)
				fs_defines.push_back("#define BLEND_MODE_MASKED");

			if (!material->is_metallic_workflow())
				fs_defines.push_back("#define SPECULAR_WORKFLOW");

			// Surface textures
			for (uint32_t i = 0; i < 6; i++)
			{
				if (material->surface_texture(i) && HAS_BIT_FLAG(flags, kRenderNodeFlags[i]))
					fs_defines.push_back(kSurfaceTextureLUT[i]);
				else if (!material->surface_texture(i) && HAS_BIT_FLAG(flags, kRenderNodeFlags[i]))
					fs_defines.push_back(kMaterialUniformLUT[i]);
			}

			// Custom Textures
			for (uint32_t i = 0; i < material->custom_texture_count(); i++)
				fs_defines.push_back(kCustomTextureLUT[i]);

			// Fragment Func
			source += m_fs_template_defines;
			source += "\n\n";
			source += m_fs_template_includes;
			source += "\n\n";

			if (material->has_fragment_shader_func())
			{
				source += "#ifndef FRAGMENT_SHADER_FUNC\n";
				source += "#define FRAGMENT_SHADER_FUNC\n";
				source += material->fragment_shader_func();
				source += "#endif\n\n";
			}
			
			source += fs_template;

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

		program->uniform_block_binding("u_PerView", 0);
		program->uniform_block_binding("u_PerEntity", 2);

		m_program_cache.set(program_key.key, program);

		return program;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble