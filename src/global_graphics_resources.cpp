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
	std::shared_ptr<VertexArray>   GlobalGraphicsResources::m_quad_vao = nullptr;
	std::shared_ptr<VertexBuffer>  GlobalGraphicsResources::m_quad_vbo = nullptr;
	std::shared_ptr<VertexArray>   GlobalGraphicsResources::m_cube_vao = nullptr;
	std::shared_ptr<VertexBuffer>  GlobalGraphicsResources::m_cube_vbo = nullptr;
	std::shared_ptr<UniformBuffer> GlobalGraphicsResources::m_per_frame = nullptr;
	std::shared_ptr<UniformBuffer> GlobalGraphicsResources::m_per_scene = nullptr;
	std::shared_ptr<UniformBuffer> GlobalGraphicsResources::m_per_entity = nullptr;
	PerFrameUniforms			   GlobalGraphicsResources::m_per_frame_uniforms;

	struct RenderTargetKey
	{
		uint32_t face = UINT32_MAX;
		uint32_t layer = UINT32_MAX;
		uint32_t mip_level = UINT32_MAX;
		uint32_t gl_id = UINT32_MAX;
	};

	struct FramebufferKey
	{
		RenderTargetKey rt_keys[8];
		RenderTargetKey depth_key;
	};

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

		// Create uniform buffers.
		m_per_frame = std::make_shared<UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(PerFrameUniforms));
		m_per_scene = std::make_shared<UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(PerSceneUniforms));
		m_per_entity = std::make_shared<UniformBuffer>(GL_DYNAMIC_DRAW, 1024 * sizeof(PerEntityUniforms));

		// Create common geometry VBO's and VAO's.
		create_quad();
		create_cube();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::shutdown()
	{
		// Delete common geometry VBO's and VAO's.
		m_quad_vao.reset();
		m_quad_vbo.reset();
		m_cube_vao.reset();
		m_cube_vbo.reset();

		// Delete uniform buffers.
		m_per_frame.reset();
		m_per_scene.reset();
		m_per_entity.reset();

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

	void GlobalGraphicsResources::initialize_render_targets(const uint32_t& window_w, const uint32_t& window_h)
	{
		// Clear all render targets
		for (int i = 0; i < m_fbo_cache.size(); i++)
		{
			NIMBLE_SAFE_DELETE(m_fbo_cache.m_value[i]);
			m_fbo_cache.remove(m_fbo_cache.m_key[i]);
		}

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

#if defined(REUSE_RENDER_TARGETS)
			// Try to find existing Render Target Textures.
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
							// If reused, make sure the depedencies don't overlap.
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

			// Else, create new texture.
			if (current->expired)
			{
				if (current->scaled)
				{
					current->w = uint32_t(current->scale_w * float(window_w));
					current->h = uint32_t(current->scale_h * float(window_h));
				}

				if (current->target == GL_TEXTURE_2D)
					current->texture = std::make_shared<Texture2D>(current->w, current->h, current->array_size, current->mip_levels, current->num_samples, current->internal_format, current->format, current->type);
				else if (current->target == GL_TEXTURE_CUBE_MAP)
					current->texture = std::make_shared<TextureCube>(current->w, current->h, current->array_size, current->mip_levels, current->internal_format, current->format, current->type);

				current->expired = false;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Framebuffer* GlobalGraphicsResources::framebuffer_for_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view)
	{
		FramebufferKey key;

		if (rt_views)
		{
			for (int i = 0; i < num_render_targets; i++)
			{
				key.rt_keys[i].face = rt_views[i].face;
				key.rt_keys[i].layer = rt_views[i].layer;
				key.rt_keys[i].mip_level = rt_views[i].mip_level;
				key.rt_keys[i].gl_id = rt_views[i].render_target->texture->id();
			}
		}
		
		if (depth_view)
		{
			key.depth_key.face = depth_view->face;
			key.depth_key.layer = depth_view->layer;
			key.depth_key.mip_level = depth_view->mip_level;
			key.depth_key.gl_id = depth_view->render_target->texture->id();;
		}

		uint64_t hash = murmur_hash_64(&key, sizeof(FramebufferKey), 5234);

		Framebuffer* fbo = nullptr;

		if (!m_fbo_cache.get(hash, fbo))
		{
			fbo = new Framebuffer();
			
			if (rt_views)
			{
				if (num_render_targets == 0)
				{
					if (rt_views[0].render_target->texture->target() == GL_TEXTURE_2D)
						fbo->attach_render_target(0, rt_views[0].render_target->texture.get(), rt_views[0].layer, rt_views[0].mip_level);
					else
						fbo->attach_render_target(0, static_cast<TextureCube*>(rt_views[0].render_target->texture.get()), rt_views[0].face, rt_views[0].layer, rt_views[0].mip_level);
				}
				else
				{
					Texture* textures[8];

					for (int i = 0; i < num_render_targets; i++)
						textures[i] = rt_views[i].render_target->texture.get();

					fbo->attach_multiple_render_targets(num_render_targets, textures);
				}
				
			}

			if (depth_view)
			{
				if (depth_view->render_target->texture->target() == GL_TEXTURE_2D)
					fbo->attach_depth_stencil_target(depth_view->render_target->texture.get(), depth_view->layer, depth_view->mip_level);
				else
					fbo->attach_depth_stencil_target(static_cast<TextureCube*>(depth_view->render_target->texture.get()), depth_view->face, depth_view->layer, depth_view->mip_level);
			}

			m_fbo_cache.set(hash, fbo);
		}

		return fbo;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::bind_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView& depth_view)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<Program> GlobalGraphicsResources::create_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs)
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

	std::shared_ptr<Program> GlobalGraphicsResources::create_program(const std::vector<std::shared_ptr<Shader>>& shaders)
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

		m_cube_vbo = std::make_shared<VertexBuffer>(GL_STATIC_DRAW, sizeof(cube_vertices), (void*)cube_vertices);

		VertexAttrib attribs[] =
		{
			{ 3,GL_FLOAT, false, 0, },
			{ 3,GL_FLOAT, false, sizeof(float) * 3 },
			{ 2,GL_FLOAT, false, sizeof(float) * 6 }
		};

		m_cube_vao = std::make_shared<VertexArray>(m_cube_vbo.get(), nullptr, sizeof(float) * 8, 3, attribs);

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

		m_quad_vbo = std::make_shared<VertexBuffer>(GL_STATIC_DRAW, sizeof(vertices), (void*)vertices);

		VertexAttrib quad_attribs[] =
		{
			{ 3, GL_FLOAT, false, 0, },
			{ 2, GL_FLOAT, false, sizeof(float) * 3 }
		};

		m_quad_vao = std::make_shared<VertexArray>(m_quad_vbo.get(), nullptr, sizeof(float) * 5, 2, quad_attribs);

		if (!m_quad_vbo || !m_quad_vao)
		{
			NIMBLE_LOG_INFO("Failed to create Vertex Buffers/Arrays");
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}