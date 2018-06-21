#include "renderer.h"
#include <camera.h>
#include <material.h>
#include <mesh.h>
#include <logger.h>
#include <utility.h>
#include <fstream>

#include "entity.h"
#include "demo_loader.h"

const float clear_color[] = { 0.1f, 0.1f, 0.1f, 1.0f };

Renderer::Renderer(uint16_t width, uint16_t height) : m_width(width), m_height(height)
{
	BufferCreateDesc per_frame_ubo_desc;
	DW_ZERO_MEMORY(per_frame_ubo_desc);
	per_frame_ubo_desc.data = nullptr;
	per_frame_ubo_desc.data_type = DataType::FLOAT;
	per_frame_ubo_desc.size = sizeof(PerFrameUniforms);
	per_frame_ubo_desc.usage_type = BufferUsageType::DYNAMIC;

	BufferCreateDesc per_entity_ubo_desc;
	DW_ZERO_MEMORY(per_entity_ubo_desc);
	per_entity_ubo_desc.data = nullptr;
	per_entity_ubo_desc.data_type = DataType::FLOAT;
	per_entity_ubo_desc.size = 1024 * sizeof(PerEntityUniforms);
	per_entity_ubo_desc.usage_type = BufferUsageType::DYNAMIC;

	BufferCreateDesc per_scene_ubo_desc;
	DW_ZERO_MEMORY(per_scene_ubo_desc);
	per_scene_ubo_desc.data = nullptr;
	per_scene_ubo_desc.data_type = DataType::FLOAT;
	per_scene_ubo_desc.size = sizeof(PerSceneUniforms);
	per_scene_ubo_desc.usage_type = BufferUsageType::DYNAMIC;

	BufferCreateDesc per_frustum_split_ubo_desc;
	DW_ZERO_MEMORY(per_frustum_split_ubo_desc);
	per_frustum_split_ubo_desc.data = nullptr;
	per_frustum_split_ubo_desc.data_type = DataType::FLOAT;
	per_frustum_split_ubo_desc.size = 256 * 8;
	per_frustum_split_ubo_desc.usage_type = BufferUsageType::DYNAMIC;

	m_per_frame = m_device->create_uniform_buffer(per_frame_ubo_desc);
	m_per_entity = m_device->create_uniform_buffer(per_entity_ubo_desc);
	m_per_scene = m_device->create_uniform_buffer(per_scene_ubo_desc);
	m_per_frustum_split = m_device->create_uniform_buffer(per_frustum_split_ubo_desc);

	Texture2DCreateDesc color_buffer_desc;
	DW_ZERO_MEMORY(color_buffer_desc);

	color_buffer_desc.height = m_height;
	color_buffer_desc.width = m_width;
	color_buffer_desc.mipmap_levels = 1;
	color_buffer_desc.format = TextureFormat::R8G8B8A8_UNORM;

	m_color_buffer = m_device->create_texture_2d(color_buffer_desc);

	Texture2DCreateDesc depth_buffer_desc;
	DW_ZERO_MEMORY(depth_buffer_desc);

	depth_buffer_desc.height = m_height;
	depth_buffer_desc.width = m_width;
	depth_buffer_desc.mipmap_levels = 1;
	depth_buffer_desc.format = TextureFormat::D32_FLOAT_S8_UINT;

	m_depth_buffer = m_device->create_texture_2d(depth_buffer_desc);

	FramebufferCreateDesc fbo_desc;
	DW_ZERO_MEMORY(fbo_desc);

	fbo_desc.depthStencilTarget.texture = m_depth_buffer;
	fbo_desc.depthStencilTarget.mipSlice = 0;
	fbo_desc.depthStencilTarget.arraySlice = 0;
	fbo_desc.renderTargetCount = 1;
	fbo_desc.renderTargets[0].texture = m_color_buffer;
	fbo_desc.renderTargets[0].arraySlice = 0;
	fbo_desc.renderTargets[0].mipSlice = 0;

	m_color_fbo = m_device->create_framebuffer(fbo_desc);

	m_brdfLUT = std::unique_ptr<dw::Texture2D>((dw::Texture2D*)demo::load_image(dw::utility::path_for_resource("assets/texture/brdfLUT.trm"), GL_RG16F, GL_RG, GL_HALF_FLOAT));

	create_cube();
	create_quad();

	// Load cubemap shaders
	{
		std::string path = dw::utility::path_for_resource("assets/shader/cubemap_vs.glsl");
		m_cube_map_vs = load_shader(GL_VERTEX_SHADER, path, nullptr);
		path = dw::utility::path_for_resource("assets/shader/cubemap_fs.glsl");
		m_cube_map_fs = load_shader(GL_FRAGMENT_SHADER, path, nullptr);

		dw::Shader* shaders[] = { m_cube_map_vs, m_cube_map_fs };

		path = dw::utility::executable_path() + "/cubemap_vs.glslcubemap_fs.glsl";
		m_cube_map_program = load_program(path, 2, &shaders[0]);

		if (!m_cube_map_vs || !m_cube_map_fs || !m_cube_map_program)
		{
			DW_LOG_ERROR("Failed to load cubemap shaders");
		}
	}

	// Load shadowmap shaders
	{
		std::string path = dw::utility::path_for_resource("assets/shader/pssm_vs.glsl");
		m_pssm_vs = load_shader(GL_VERTEX_SHADER, path, nullptr);
		path = dw::utility::path_for_resource("assets/shader/pssm_fs.glsl");
		m_pssm_fs = load_shader(GL_FRAGMENT_SHADER, path, nullptr);

		dw::Shader* shaders[] = { m_pssm_vs, m_pssm_fs };

		path = dw::utility::executable_path() + "/pssm_vs.glslpssm_fs.glsl";
		m_pssm_program = load_program(path, 2, &shaders[0]);

		if (!m_pssm_vs || !m_pssm_fs || !m_pssm_program)
		{
			DW_LOG_ERROR("Failed to load PSSM shaders");
		}
	}

	m_per_scene_uniforms.pointLightCount = 0;
	m_per_scene_uniforms.pointLights[0].position = glm::vec4(-10.0f, 20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[0].color = glm::vec4(300.0f);
	m_per_scene_uniforms.pointLights[1].position = glm::vec4(10.0f, 20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[1].color = glm::vec4(300.0f);
	m_per_scene_uniforms.pointLights[2].position = glm::vec4(-10.0f, -20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[2].color = glm::vec4(300.0f);
	m_per_scene_uniforms.pointLights[3].position = glm::vec4(10.0f, -20.0f, 10.0f, 1.0f);
	m_per_scene_uniforms.pointLights[3].color = glm::vec4(300.0f);

	m_per_scene_uniforms.directionalLight.color = glm::vec4(1.0f);
	m_per_scene_uniforms.directionalLight.direction = glm::vec4(glm::normalize(glm::vec3(1.0f, -1.0f, 0.0f)), 1.0f);
}

Renderer::~Renderer()
{
	for (auto itr : m_program_cache)
	{
		DW_SAFE_DELETE(itr.second);
	}

	for (auto itr : m_shader_cache)
	{
		DW_SAFE_DELETE(itr.second);
	}
}

void Renderer::create_cube()
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

	m_cube_vbo = std::make_unique<dw::VertexBuffer>(GL_STATIC_DRAW, sizeof(cube_vertices), cube_vertices);

	dw::VertexAttrib attribs[] =
	{
		{ 3,GL_FLOAT, false, 0,				   },
		{ 3,GL_FLOAT, false, sizeof(float) * 3 },
		{ 2,GL_FLOAT, false, sizeof(float) * 6 }
	};

	m_cube_vao = std::make_unique<dw::VertexArray>(m_cube_vbo.get(), nullptr, sizeof(float) * 8, 3, attribs);

	if (!m_cube_vbo || !m_cube_vao)
	{
		DW_LOG_FATAL("Failed to create Vertex Buffers/Arrays");
	}
}

void Renderer::create_quad()
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

	m_quad_vbo = std::make_unique<dw::VertexBuffer>(GL_STATIC_DRAW, sizeof(vertices), vertices);

	dw::VertexAttrib quad_attribs[] =
	{
		{ 3, GL_FLOAT, false, 0,			    },
		{ 2, GL_FLOAT, false, sizeof(float) * 3 }
	};

	m_quad_vao = std::make_unique<dw::VertexArray>(m_quad_vbo.get(), nullptr, sizeof(float) * 5, 2, quad_attribs);

	if (!m_quad_vbo || !m_quad_vao)
	{
		DW_LOG_INFO("Failed to create Vertex Buffers/Arrays");
	}

	// Load quad shaders
	{
		std::string path = dw::utility::executable_path() + "/assets/shader/quad_vs.glsl";
		m_quad_vs = load_shader(GL_VERTEX_SHADER, path, nullptr);
		path = dw::utility::executable_path() + "/assets/shader/quad_fs.glsl";
		m_quad_fs = load_shader(GL_FRAGMENT_SHADER, path, nullptr);

		dw::Shader* shaders[] = { m_quad_vs, m_quad_fs };

		path = dw::utility::executable_path() + "/quad_vs.glslquadfs.glsl";
		m_quad_program = load_program(path, 2, &shaders[0]);

		if (!m_quad_vs || !m_quad_fs || !m_quad_program)
		{
			DW_LOG_INFO("Failed to load Quad shaders");
		}
	}
}

void Renderer::set_scene(Scene* scene)
{
	m_scene = scene;
}

Scene* Renderer::scene()
{
	return m_scene;
}

dw::Shader* Renderer::load_shader(GLuint type, std::string& path, dw::Material* mat)
{
	if (m_shader_cache.find(path) == m_shader_cache.end())
	{
		DW_LOG_INFO("Shader Asset not in cache. Loading from disk.");

		dw::Shader* shader = dw::Shader::create_from_file(type, dw::utility::path_for_resource("assets/" + path));
		m_shader_cache[path] = shader;
		return shader;
	}
	else
	{
		DW_LOG_INFO("Shader Asset already loaded. Retrieving from cache.");
		return m_shader_cache[path];
	}
}

dw::Program* Renderer::load_program(std::string& combined_name, uint32_t count, dw::Shader** shaders)
{
	if (m_program_cache.find(combined_name) == m_program_cache.end())
	{
		DW_LOG_INFO("Shader Program Asset not in cache. Loading from disk.");

		dw::Program* program = new dw::Program(count, shaders);
		m_program_cache[combined_name] = program;

		return program;
	}
	else
	{
		DW_LOG_INFO("Shader Program Asset already loaded. Retrieving from cache.");
		return m_program_cache[combined_name];
	}
}

void Renderer::render(dw::Camera* camera, uint16_t w, uint16_t h, dw::Framebuffer* fbo)
{
	Entity** entities = m_scene->entities();
	int entity_count = m_scene->entity_count();

	m_per_frame_uniforms.projMat = camera->m_projection;
	m_per_frame_uniforms.viewMat = camera->m_view;
	m_per_frame_uniforms.viewProj = camera->m_view_projection;
	m_per_frame_uniforms.viewDir = glm::vec4(camera->m_forward.x, camera->m_forward.y, camera->m_forward.z, 0.0f);
	m_per_frame_uniforms.viewPos = glm::vec4(camera->m_position.x, camera->m_position.y, camera->m_position.z, 0.0f);
	//m_per_frame_uniforms.numCascades = shadows->frustum_split_count();
	//memcpy(&m_per_frame_uniforms.shadowFrustums[0], shadows->frustum_splits(), sizeof(ShadowFrustum) * m_per_frame_uniforms.numCascades);

	for (int i = 0; i < entity_count; i++)
	{
		Entity* entity = entities[i];
		m_per_entity_uniforms[i].modalMat = entity->m_transform;
		m_per_entity_uniforms[i].mvpMat = camera->m_view_projection * entity->m_transform;
		m_per_entity_uniforms[i].worldPos = glm::vec4(entity->m_position.x, entity->m_position.y, entity->m_position.z, 0.0f);
	}

	void* mem = m_per_frame->map(GL_WRITE_ONLY);

	if (mem)
	{
		memcpy(mem, &m_per_frame_uniforms, sizeof(PerFrameUniforms));
		m_per_frame->unmap();
	}

	mem = m_per_scene->map(GL_WRITE_ONLY);

	if (mem)
	{
		memcpy(mem, &m_per_scene_uniforms, sizeof(PerSceneUniforms));
		m_per_scene->unmap();
	}

	mem = m_per_entity->map(GL_WRITE_ONLY);

	if (mem)
	{
		memcpy(mem, &m_per_entity_uniforms[0], sizeof(PerEntityUniforms) * entity_count);
		m_per_entity->unmap();
	}

	render_scene(w, h, fbo);
}

void Renderer::render_scene(uint16_t w, uint16_t h, dw::Framebuffer* fbo)
{
	// Bind framebuffer.
	if (fbo)
		fbo->bind();
	else
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	// Set viewport and clear framebuffer.
	glViewport(0, 0, w, h);
	glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	Entity** entities = m_scene->entities();
	int entity_count = m_scene->entity_count();

	for (int i = 0; i < entity_count; i++)
	{
		Entity* entity = entities[i];

		if (!entity->m_mesh || !entity->m_program)
			continue;

		// Bind program.
		dw::Program* current_program = entity->m_program;

		current_program->use();

		// Set rasterizer and depth states.
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		// Bind uniform buffers.
		m_per_frame->bind_base(0);
		m_per_scene->bind_base(2);

		dw::SubMesh* submeshes = entity->m_mesh->sub_meshes();

		// Bind environment textures.
		m_scene->irradiance_map()->bind(4);
		current_program->set_uniform("s_IrradianceMap", 4);

		m_scene->prefiltered_map()->bind(5);
		current_program->set_uniform("s_PrefilteredMap", 5);

		m_brdfLUT->bind(6);
		current_program->set_uniform("s_BRDF", 6);

		for (uint32_t j = 0; j < entity->m_mesh->sub_mesh_count(); j++)
		{
			dw::Material* mat = submeshes[j].mat;

			if (!mat)
				mat = entity->m_override_mat;

			// Bind vertex array.
			entity->m_mesh->mesh_vertex_array()->bind();

			// Bind materials.
			if (mat)
			{
				dw::Texture2D* albedo = mat->texture(TEXTURE_ALBEDO);

				if (albedo)
				{
					albedo->bind(0);
					current_program->set_uniform("s_Albedo", 0);
				}

				dw::Texture2D* normal = mat->texture(TEXTURE_NORMAL);

				if (normal)
				{
					normal->bind(0);
					current_program->set_uniform("s_Normal", 1);
				}

				dw::Texture2D* metalness = mat->texture(TEXTURE_METALNESS);

				if (metalness)
				{
					metalness->bind(2);
					current_program->set_uniform("s_Metalness", 2);
				}

				dw::Texture2D* roughness = mat->texture(TEXTURE_ROUGHNESS);

				if (roughness)
				{
					roughness->bind(3);
					current_program->set_uniform("s_Roughness", 3);
				}

				dw::Texture2D* displacement = mat->texture(TEXTURE_DISPLACEMENT);

				if (displacement)
				{
					displacement->bind(4);
					current_program->set_uniform("s_Displacement", 4);
				}

				dw::Texture2D* emissive = mat->texture(TEXTURE_EMISSIVE);

				if (emissive)
				{
					emissive->bind(5);
					current_program->set_uniform("s_Emissive", 5);
				}
			}

			// Bind per-entity uniforms.
			m_per_entity->bind_range(1, i * sizeof(PerEntityUniforms), sizeof(PerEntityUniforms));

			// Issue draw call.
			glDrawElementsBaseVertex(GL_TRIANGLES, submeshes[j].index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * submeshes[j].base_index), submeshes[j].base_vertex);
		}
	}
}