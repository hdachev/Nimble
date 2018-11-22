#include "global_graphics_resources.h"
#include "demo_loader.h"
#include "uniforms.h"
#include "constants.h"
#include "utility.h"
#include "logger.h"
#include "macros.h"

namespace nimble
{
	std::unordered_map<std::string, Texture*> GlobalGraphicsResources::m_texture_map;
	std::unordered_map<std::string, Framebuffer*> GlobalGraphicsResources::m_framebuffer_map;
	std::unordered_map<std::string, Program*> GlobalGraphicsResources::m_program_cache;
	std::unordered_map<std::string, Shader*> GlobalGraphicsResources::m_shader_cache;
	VertexArray*   GlobalGraphicsResources::m_quad_vao = nullptr;
	VertexBuffer*  GlobalGraphicsResources::m_quad_vbo = nullptr;
	VertexArray*   GlobalGraphicsResources::m_cube_vao = nullptr;
	VertexBuffer*  GlobalGraphicsResources::m_cube_vbo = nullptr;
	UniformBuffer* GlobalGraphicsResources::m_per_frame = nullptr;
	UniformBuffer* GlobalGraphicsResources::m_per_scene = nullptr;
	UniformBuffer* GlobalGraphicsResources::m_per_entity = nullptr;
	PerFrameUniforms   GlobalGraphicsResources::m_per_frame_uniforms;

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::initialize()
	{
		// Set initial settings.
		m_per_frame_uniforms.ssao = 1;
		m_per_frame_uniforms.motion_blur = 1;
		m_per_frame_uniforms.renderer = 1;
		m_per_frame_uniforms.current_output = SHOW_DEFERRED_COLOR;
		m_per_frame_uniforms.max_motion_blur_samples = 32;
		m_per_frame_uniforms.ssao_num_samples = 64;
		m_per_frame_uniforms.ssao_radius = 10.0f;
		m_per_frame_uniforms.ssao_bias = 0.025f;

		// Load BRDF look-up-texture.
		Texture* brdf_lut = demo::load_image("texture/brdfLUT.trm", GL_RG16F, GL_RG, GL_HALF_FLOAT);
		brdf_lut->set_min_filter(GL_LINEAR);
		brdf_lut->set_mag_filter(GL_LINEAR);

		m_texture_map[BRDF_LUT] = brdf_lut;

		// Create uniform buffers.
		m_per_frame = new UniformBuffer(GL_DYNAMIC_DRAW, sizeof(PerFrameUniforms));
		m_per_scene = new UniformBuffer(GL_DYNAMIC_DRAW, sizeof(PerSceneUniforms));
		m_per_entity = new UniformBuffer(GL_DYNAMIC_DRAW, 1024 * sizeof(PerEntityUniforms));

		// Create common geometry VBO's and VAO's.
		create_quad();
		create_cube();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::shutdown()
	{
		// Delete common geometry VBO's and VAO's.
		NIMBLE_SAFE_DELETE(m_quad_vao);
		NIMBLE_SAFE_DELETE(m_quad_vbo);
		NIMBLE_SAFE_DELETE(m_cube_vao);
		NIMBLE_SAFE_DELETE(m_cube_vbo);

		// Delete uniform buffers.
		NIMBLE_SAFE_DELETE(m_per_frame);
		NIMBLE_SAFE_DELETE(m_per_scene);
		NIMBLE_SAFE_DELETE(m_per_entity);

		// Delete framebuffers.
		for (auto itr : m_framebuffer_map)
		{
			NIMBLE_SAFE_DELETE(itr.second);
		}

		// Delete textures.
		for (auto itr : m_texture_map)
		{
			NIMBLE_SAFE_DELETE(itr.second);
		}

		// Delete programs.
		for (auto itr : m_program_cache)
		{
			NIMBLE_SAFE_DELETE(itr.second);
		}

		// Delete shaders.
		for (auto itr : m_shader_cache)
		{
			NIMBLE_SAFE_DELETE(itr.second);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Texture* GlobalGraphicsResources::lookup_texture(const std::string& name)
	{
		if (m_texture_map.find(name) == m_texture_map.end())
			return nullptr;
		else
			return m_texture_map[name];
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> GlobalGraphicsResources::request_render_target(const uint32_t& graph_id, const uint32_t& node_id, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

		rt->graph_id = graph_id;
		rt->node_id = node_id;
		rt->last_dependent_node_id = node_id;
		rt->expired = true;
		rt->scaled = false;
		rt->w = w;
		rt->h = h;
		rt->scale_w = 0.0f;
		rt->scale_h = 0.0f;
		rt->internal_format = internal_format;
		rt->format = format;
		rt->type = type;
		rt->num_samples = num_samples;
		rt->array_size = array_size;
		rt->mip_levels = mip_levels;

		m_render_target_pool.push_back(rt);

		return rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> GlobalGraphicsResources::request_scaled_render_target(const uint32_t& graph_id, const uint32_t& node_id, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

		rt->graph_id = graph_id;
		rt->node_id = node_id;
		rt->last_dependent_node_id = node_id;
		rt->expired = true;
		rt->scaled = true;
		rt->w = 0;
		rt->h = 0;
		rt->scale_w = w;
		rt->scale_h = h;
		rt->internal_format = internal_format;
		rt->format = format;
		rt->type = type;
		rt->num_samples = num_samples;
		rt->array_size = array_size;
		rt->mip_levels = mip_levels;

		m_render_target_pool.push_back(rt);

		return rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::initialize_render_targets()
	{
		for (auto& rt : m_render_target_pool)
		{
			std::shared_ptr<RenderTarget> rt_ptr = rt.lock();

			if (rt_ptr)
			{
				rt_ptr->expired = true;
				rt_ptr->texture.reset();
			}
		}

		for (uint32_t i = 0; i < m_render_target_pool.size(); i++)
		{
			auto& current = m_render_target_pool[i].lock();

			// Try to find existing Render Target Texture
			for (uint32_t j = 0; j < m_render_target_pool.size(); j++)
			{
				if (m_render_target_pool[j].expired() || i == j)
					continue;
				else
				{
					const auto& current_inner = m_render_target_pool[j].lock();

					if (current->scaled == current_inner->scaled &&
						current->w == current_inner->w &&
						current->h == current_inner->h &&
						current->scale_w == current_inner->scale_w &&
						current->scale_h == current_inner->scale_h &&
						current->node_id != current_inner->node_id &&
						!current_inner->expired)
					{
						if ((current->graph_id != current_inner->graph_id) || (current->graph_id == current_inner->graph_id && current->node_id > current_inner->last_dependent_node_id))
						{
							// Make sure the texture isn't re-used already
							bool reused = false;

							for (uint32_t k = 0; k < m_render_target_pool.size(); k++)
							{
								if (k == i || k == j)
									continue;

								const auto& rt = m_render_target_pool[k].lock();

								if (rt && rt->texture->id() == current_inner->texture->id() && 
									((rt->node_id > current->node_id && 
									  rt->node_id < current->last_dependent_node_id) || 
									 (rt->last_dependent_node_id > current->node_id && 
									  rt->last_dependent_node_id < current->last_dependent_node_id) || 
									  rt->node_id == current->node_id || 
									  rt->node_id == current->last_dependent_node_id ||
									  rt->last_dependent_node_id == current->node_id ||
									  rt->last_dependent_node_id == current->last_dependent_node_id))
								{
									reused = true;
									break;
								}
							}

							if (!reused)
							{
								current->texture = current_inner->texture;
								current->expired = false;

								break;
							}
						}
					}
				}
			}

			// Else, create new texture
			if (current->expired)
			{
				current->expired = false;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Texture2D* GlobalGraphicsResources::create_texture_2d(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		if (m_texture_map.find(name) == m_texture_map.end())
		{
			Texture2D* texture = new Texture2D(w, h, array_size, mip_levels, num_samples, internal_format, format, type);
			m_texture_map[name] = texture;

			return texture;
		}
		else
		{
			NIMBLE_LOG_ERROR("A texture with the requested name (" + name + ") already exists. Returning nullptr...");
			return nullptr;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	TextureCube* GlobalGraphicsResources::create_texture_cube(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t array_size, uint32_t mip_levels)
	{
		if (m_texture_map.find(name) == m_texture_map.end())
		{
			TextureCube* texture = new TextureCube(w, h, array_size, mip_levels, internal_format, format, type);
			m_texture_map[name] = texture;

			return texture;
		}
		else
		{
			NIMBLE_LOG_ERROR("A texture with the requested name (" + name + ") already exists. Returning nullptr...");
			return nullptr;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::destroy_texture(const std::string& name)
	{
		if (m_texture_map.find(name) != m_texture_map.end())
		{
			Texture* texture = m_texture_map[name];
			NIMBLE_SAFE_DELETE(texture);
			m_texture_map.erase(name);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Framebuffer* GlobalGraphicsResources::lookup_framebuffer(const std::string& name)
	{
		if (m_framebuffer_map.find(name) == m_framebuffer_map.end())
			return nullptr;
		else
			return m_framebuffer_map[name];
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Framebuffer* GlobalGraphicsResources::create_framebuffer(const std::string& name)
	{
		if (m_framebuffer_map.find(name) == m_framebuffer_map.end())
		{
			Framebuffer* fbo = new Framebuffer();
			m_framebuffer_map[name] = fbo;

			return fbo;
		}
		else
		{
			NIMBLE_LOG_ERROR("A framebuffer with the requested name (" + name + ") already exists. Returning nullptr...");
			return nullptr;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::destroy_framebuffer(const std::string& name)
	{
		if (m_framebuffer_map.find(name) != m_framebuffer_map.end())
		{
			Framebuffer* fbo = m_framebuffer_map[name];
			NIMBLE_SAFE_DELETE(fbo);
			m_framebuffer_map.erase(name);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Shader* GlobalGraphicsResources::load_shader(GLuint type, std::string& path, const std::vector<std::string> defines)
	{
		std::string name_with_defines = "";

		for (auto define : defines)
		{
			name_with_defines += define;
			name_with_defines += "|";
		}

		name_with_defines += path;

		if (m_shader_cache.find(name_with_defines) == m_shader_cache.end())
		{
			std::string source;

			if (!read_shader(utility::path_for_resource("assets/" + path), source, defines))
			{
				NIMBLE_LOG_ERROR("Failed to read shader with name '" + path);
				return nullptr;
			}

			Shader* shader = new Shader(type, source);

			if (!shader->compiled())
			{
				NIMBLE_LOG_ERROR("Shader with name '" + path + "' failed to compile:\n" + source);
				NIMBLE_SAFE_DELETE(shader);
				return nullptr;
			}

			m_shader_cache[name_with_defines] = shader;
			return shader;
		}
		else
			return m_shader_cache[name_with_defines];
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Program* GlobalGraphicsResources::load_program(std::string& combined_name, uint32_t count, Shader** shaders)
	{
		if (m_program_cache.find(combined_name) == m_program_cache.end())
		{
			Program* program = new Program(count, shaders);

			if (!program)
			{
				NIMBLE_LOG_ERROR("Program with combined name '" + combined_name + "' failed to link!");
				return nullptr;
			}

			m_program_cache[combined_name] = program;

			return program;
		}
		else
			return m_program_cache[combined_name];
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string GlobalGraphicsResources::combined_program_name(const std::string& vs, const std::string& fs, std::vector<std::string> defines)
	{
		std::string name_with_defines = "";

		for (auto define : defines)
		{
			name_with_defines += define;
			name_with_defines += "|";
		}

		name_with_defines += vs;
		name_with_defines += fs;

		return name_with_defines;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::create_cube()
	{
		float cube_vertices[] =
		{
			// back face
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
			1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			-1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
			// front face
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
			1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			-1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			// left face
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			-1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			// right face
			1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
			1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
			// bottom face
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
			1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			-1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			// top face
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
			1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			-1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
		};

		m_cube_vbo = new VertexBuffer(GL_STATIC_DRAW, sizeof(cube_vertices), (void*)cube_vertices);

		VertexAttrib attribs[] =
		{
			{ 3,GL_FLOAT, false, 0, },
			{ 3,GL_FLOAT, false, sizeof(float) * 3 },
			{ 2,GL_FLOAT, false, sizeof(float) * 6 }
		};

		m_cube_vao = new VertexArray(m_cube_vbo, nullptr, sizeof(float) * 8, 3, attribs);

		if (!m_cube_vbo || !m_cube_vao)
		{
			NIMBLE_LOG_FATAL("Failed to create Vertex Buffers/Arrays");
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::create_quad()
	{
		const float vertices[] =
		{
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f
		};

		m_quad_vbo = new VertexBuffer(GL_STATIC_DRAW, sizeof(vertices), (void*)vertices);

		VertexAttrib quad_attribs[] =
		{
			{ 3, GL_FLOAT, false, 0, },
			{ 2, GL_FLOAT, false, sizeof(float) * 3 }
		};

		m_quad_vao = new VertexArray(m_quad_vbo, nullptr, sizeof(float) * 5, 2, quad_attribs);

		if (!m_quad_vbo || !m_quad_vao)
		{
			NIMBLE_LOG_INFO("Failed to create Vertex Buffers/Arrays");
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	template<typename T>
	bool contains(const std::vector<T>& vec, const T& obj)
	{
		for (auto& e : vec)
		{
			if (e == obj)
				return true;
		}

		return false;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool GlobalGraphicsResources::read_shader(std::string path, std::string& out, const std::vector<std::string> defines)
	{
		std::string og_source;

		if (!utility::read_text(path, og_source))
			return false;

		if (defines.size() > 0)
		{
			for (auto define : defines)
				out += "#define " + define + "\n";

			out += "\n";
		}

		std::istringstream stream(og_source);
		std::string line;
		std::vector<std::string> included_headers;

		while (std::getline(stream, line))
		{
			if (line.find("#include") != std::string::npos)
			{
				size_t start = line.find_first_of("<") + 1;
				size_t end = line.find_last_of(">");
				std::string include_path = line.substr(start, end - start);

				std::string path_to_shader = "";
				size_t slash_pos = path.find_last_of("/");

				if (slash_pos != std::string::npos)
					path_to_shader = path.substr(0, slash_pos + 1);

				std::string include_source;

				if (!read_shader(path_to_shader + include_path, include_source, std::vector<std::string>()))
				{
					NIMBLE_LOG_ERROR("Included file <" + include_path + "> cannot be opened!");
					return false;
				}
				if (contains(included_headers, include_path))
					NIMBLE_LOG_WARNING("Header <" + include_path + "> has been included twice!");
				else
				{
					included_headers.push_back(include_path);
					out += include_source + "\n\n";
				}
			}
			else
				out += line + "\n";
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}