#include "global_graphics_resources.h"
#include "demo_loader.h"
#include "uniforms.h"
#include "constants.h"
#include "utility.h"
#include "logger.h"
#include "macros.h"

namespace nimble
{
	std::vector<std::weak_ptr<RenderTarget>> GlobalGraphicsResources::m_render_target_pool;
	std::unordered_map<std::string, std::weak_ptr<Program>> GlobalGraphicsResources::m_program_cache;
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

		// Delete render targets.
		for (auto itr : m_render_target_pool)
			itr.reset();

		// Delete programs.
		for (auto itr : m_program_cache)
			itr.second.reset();
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

#if defined(REUSED_RENDER_TARGETS)
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
						current->target == current_inner->target &&
						current->internal_format == current_inner->internal_format &&
						current->format == current_inner->format &&
						current->type == current_inner->type &&
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
#endif

			// Else, create new texture
			if (current->expired)
			{
				if (current->target == GL_TEXTURE_2D)
					current->texture = std::make_shared<Texture2D>(current->w, current->h, current->array_size, current->mip_levels, current->num_samples, current->internal_format, current->format, current->type);
				else if (current->target == GL_TEXTURE_CUBE_MAP)
					current->texture = std::make_shared<TextureCube>(current->w, current->h, current->array_size, current->mip_levels, current->internal_format, current->format, current->type);

				current->expired = false;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<Program> GlobalGraphicsResources::load_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs)
	{
		std::string id = std::to_string(vs->id()) + "-";
		id += std::to_string(fs->id());

		if (m_program_cache.find(id) != m_program_cache.end() && m_program_cache[id].lock())
			return m_program_cache[id].lock();
		else
		{
			Shader* shaders[] = { vs.get(), fs.get() };

			std::shared_ptr<Program> program = std::make_shared<Program>(2, shaders);

			if (!program)
			{
				NIMBLE_LOG_ERROR("Program failed to link!");
				return nullptr;
			}

			m_program_cache[id] = program;

			return program;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<Program> GlobalGraphicsResources::load_program(const std::vector<std::shared_ptr<Shader>>& shaders)
	{
		std::vector<Shader*> shaders_raw;
		std::string id = "";

		for (const auto& shader : shaders)
		{
			shaders_raw.push_back(shader.get());
			id += std::to_string(shader->id());
			id += "-";
		}

		if (m_program_cache.find(id) != m_program_cache.end() && m_program_cache[id].lock())
			return m_program_cache[id].lock();
		else
		{
			std::shared_ptr<Program> program = std::make_shared<Program>(shaders_raw.size(), shaders_raw.data());

			if (!program)
			{
				NIMBLE_LOG_ERROR("Program failed to link!");
				return nullptr;
			}

			m_program_cache[id] = program;

			return program;
		}
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