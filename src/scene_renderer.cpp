#include "scene_renderer.h"
#include "scene.h"
#include "mesh.h"
#include "global_graphics_resources.h"
#include "uniforms.h"
#include "constants.h"

// -----------------------------------------------------------------------------------------------------------------------------------

SceneRenderer::SceneRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

SceneRenderer::~SceneRenderer() {}

// -----------------------------------------------------------------------------------------------------------------------------------

void SceneRenderer::render(Scene* scene, dw::Framebuffer* fbo, dw::Program* global_program, uint32_t w, uint32_t h, uint32_t clear_flags, float* clear_color, uint32_t tex_type_count, uint32_t* tex_types, MeshRenderCallback render_callback)
{
	dw::Program* current_program = nullptr;

	// If a global program is provided, use it for further use instead.
	if (global_program)
		current_program = global_program;

	// Bind framebuffer.
	if (fbo)
		fbo->bind();
	else
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Set and clear viewport.
	glViewport(0, 0, w, h);

	if (clear_color)
		glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);

	glClear(clear_flags);

	// Bind global UBO's.
	GlobalGraphicsResources::per_frame_ubo()->bind_base(0);
	GlobalGraphicsResources::per_scene_ubo()->bind_base(2);

	Entity** entities = scene->entities();

	for (uint32_t i = 0; i < scene->entity_count(); i++)
	{
		Entity* entity = entities[i];

		// Bind entity specific uniform buffer range.
		GlobalGraphicsResources::per_entity_ubo()->bind_range(1, i * sizeof(PerEntityUniforms), sizeof(PerEntityUniforms));

		// Bind entity-specific program if no global program is provided.
		if (!global_program)
		{
			current_program = entity->m_program;
			current_program->use();
		}

		// Call mesh render callback, if given.
		if (render_callback)
			render_callback(current_program);

		// Bind environment textures.
		if (tex_type_count == ALL_TEXTURES)
		{
			scene->irradiance_map()->bind(4);
			current_program->set_uniform("s_IrradianceMap", 4);

			scene->prefiltered_map()->bind(5);
			current_program->set_uniform("s_PrefilteredMap", 5);

			dw::Texture* brdf_lut = GlobalGraphicsResources::lookup_texture(BRDF_LUT);
			brdf_lut->bind(6);
			current_program->set_uniform("s_BRDF", 6);
		}

		// Bind vertex array.
		entity->m_mesh->mesh_vertex_array()->bind();

		dw::SubMesh* submeshes = entity->m_mesh->sub_meshes();

		for (uint32_t sub_mesh_id = 0; sub_mesh_id < entity->m_mesh->sub_mesh_count(); sub_mesh_id++)
		{
			dw::SubMesh& sub_mesh = submeshes[sub_mesh_id];

			// Bind material textures, if available.
			dw::Material* mat = nullptr;

			if (entity->m_override_mat)
				mat = entity->m_override_mat;
			else
				mat = sub_mesh.mat;

			if (tex_type_count == ALL_TEXTURES)
			{
				for (uint32_t texture_idx = 0; texture_idx < m_texture_count; texture_idx++)
				{
					dw::Texture* texture = mat->texture(texture_idx);

					if (texture)
					{
						texture->bind(m_texture_flags[texture_idx]);
						current_program->set_uniform(m_texture_uniform_names[texture_idx], m_texture_flags[texture_idx]);
					}
				}
			}
			else if (tex_types && sub_mesh.mat)
			{
				// Check if this texture type is among the list of provided texture types. 
				for (uint32_t requested_type_idx = 0; requested_type_idx < tex_type_count; requested_type_idx++)
				{
					for (uint32_t texture_idx = 0; texture_idx < m_texture_count; texture_idx++)
					{
						dw::Texture* texture = mat->texture(texture_idx);

						if (m_texture_flags[texture_idx] == tex_types[requested_type_idx] && texture)
						{
							texture->bind(m_texture_flags[texture_idx]);
							current_program->set_uniform(m_texture_uniform_names[texture_idx], m_texture_flags[texture_idx]);
						}
					}
				}
			}

			// Issue draw call.
			glDrawElementsBaseVertex(GL_TRIANGLES, sub_mesh.index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * sub_mesh.base_index), sub_mesh.base_vertex);
		}
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------
