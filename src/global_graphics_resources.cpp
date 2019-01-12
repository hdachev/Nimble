#include "global_graphics_resources.h"
#include "uniforms.h"
#include "constants.h"
#include "utility.h"
#include "logger.h"
#include "macros.h"

namespace nimble
{
	std::unordered_map<std::string, std::weak_ptr<Program>> GlobalGraphicsResources::m_program_cache;
	StaticHashMap<uint64_t, Framebuffer*, 1024> GlobalGraphicsResources::m_fbo_cache;
	std::shared_ptr<VertexArray>   GlobalGraphicsResources::m_cube_vao = nullptr;
	std::shared_ptr<VertexBuffer>  GlobalGraphicsResources::m_cube_vbo = nullptr;
	std::unique_ptr<UniformBuffer> GlobalGraphicsResources::m_per_view = nullptr;
	std::unique_ptr<UniformBuffer> GlobalGraphicsResources::m_per_entity = nullptr;
	std::unique_ptr<ShaderStorageBuffer> GlobalGraphicsResources::m_per_scene = nullptr;

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
		// Create uniform buffers.
		m_per_view = std::make_unique<UniformBuffer>(GL_DYNAMIC_DRAW, MAX_VIEWS * sizeof(PerViewUniforms));
		m_per_entity = std::make_unique<UniformBuffer>(GL_DYNAMIC_DRAW, MAX_ENTITIES * sizeof(PerEntityUniforms));
		m_per_scene = std::make_unique<ShaderStorageBuffer>(GL_DYNAMIC_DRAW, sizeof(PerSceneUniforms));

		// Create common geometry VBO's and VAO's.
		create_cube();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::shutdown()
	{
		// Delete common geometry VBO's and VAO's.
		m_cube_vao.reset();
		m_cube_vbo.reset();

		// Delete uniform buffers.
		m_per_view.reset();
		m_per_entity.reset();
		m_per_scene.reset();

		// Delete framebuffer
		for (int i = 0; i < m_fbo_cache.size(); i++)
		{
			NIMBLE_SAFE_DELETE(m_fbo_cache.m_value[i]);
			m_fbo_cache.remove(m_fbo_cache.m_key[i]);
		}

		// Delete programs.
		for (auto itr : m_program_cache)
			itr.second.reset();
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
				key.rt_keys[i].gl_id = rt_views[i].texture->id();
			}
		}
		
		if (depth_view)
		{
			key.depth_key.face = depth_view->face;
			key.depth_key.layer = depth_view->layer;
			key.depth_key.mip_level = depth_view->mip_level;
			key.depth_key.gl_id = depth_view->texture->id();;
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
					if (rt_views[0].texture->target() == GL_TEXTURE_2D)
						fbo->attach_render_target(0, rt_views[0].texture.get(), rt_views[0].layer, rt_views[0].mip_level);
					else if (rt_views[0].texture->target() == GL_TEXTURE_CUBE_MAP)
						fbo->attach_render_target(0, static_cast<TextureCube*>(rt_views[0].texture.get()), rt_views[0].face, rt_views[0].layer, rt_views[0].mip_level);
				}
				else
				{
					Texture* textures[8];

					for (int i = 0; i < num_render_targets; i++)
						textures[i] = rt_views[i].texture.get();

					fbo->attach_multiple_render_targets(num_render_targets, textures);
				}
				
			}

			if (depth_view)
			{
				if (depth_view->texture->target() == GL_TEXTURE_2D)
					fbo->attach_depth_stencil_target(depth_view->texture.get(), depth_view->layer, depth_view->mip_level);
				else if (depth_view->texture->target() == GL_TEXTURE_CUBE_MAP)
					fbo->attach_depth_stencil_target(static_cast<TextureCube*>(depth_view->texture.get()), depth_view->face, depth_view->layer, depth_view->mip_level);
			}

			m_fbo_cache.set(hash, fbo);
		}

		return fbo;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GlobalGraphicsResources::bind_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view)
	{
		Framebuffer* fbo = framebuffer_for_render_targets(num_render_targets, rt_views, depth_view);

		if (fbo)
			fbo->bind();
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

			program->uniform_block_binding("u_PerView", 0);
			program->uniform_block_binding("u_PerScene", 1);
			program->uniform_block_binding("u_PerEntity", 2);

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

			program->uniform_block_binding("u_PerView", 0);
			program->uniform_block_binding("u_PerScene", 1);
			program->uniform_block_binding("u_PerEntity", 2);

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
}