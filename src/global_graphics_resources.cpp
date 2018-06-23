#include "global_graphics_resources.h"
#include "demo_loader.h"
#include "uniforms.h"
#include "constants.h"

#include <logger.h>
#include <macros.h>

std::unordered_map<std::string, dw::Texture*> GlobalGraphicsResources::m_texture_map;
std::unordered_map<std::string, dw::Framebuffer*> GlobalGraphicsResources::m_framebuffer_map;
dw::UniformBuffer* GlobalGraphicsResources::m_per_frame = nullptr;
dw::UniformBuffer* GlobalGraphicsResources::m_per_scene = nullptr;
dw::UniformBuffer* GlobalGraphicsResources::m_per_entity = nullptr;

// -----------------------------------------------------------------------------------------------------------------------------------

void GlobalGraphicsResources::initialize()
{
	// Load BRDF look-up-texture.
	dw::Texture* brdf_lut = demo::load_image("texture/brdfLUT.trm", GL_RG16F, GL_RG, GL_HALF_FLOAT);
	brdf_lut->set_min_filter(GL_LINEAR);
	brdf_lut->set_mag_filter(GL_LINEAR);

	m_texture_map[BRDF_LUT] = brdf_lut;

	// Create uniform buffers.
	m_per_frame = new dw::UniformBuffer(GL_DYNAMIC_DRAW, sizeof(PerFrameUniforms));
	m_per_scene = new dw::UniformBuffer(GL_DYNAMIC_DRAW, sizeof(PerSceneUniforms));
	m_per_entity = new dw::UniformBuffer(GL_DYNAMIC_DRAW, 1024 * sizeof(PerEntityUniforms));
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GlobalGraphicsResources::shutdown()
{
	// Delete uniform buffers.
	DW_SAFE_DELETE(m_per_frame);
	DW_SAFE_DELETE(m_per_scene);
	DW_SAFE_DELETE(m_per_entity);

	// Delete framebuffers.
	for (auto pair : m_framebuffer_map)
	{
		DW_SAFE_DELETE(pair.second);
	}

	// Delete textures.
	for (auto pair : m_texture_map)
	{
		DW_SAFE_DELETE(pair.second);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::Texture* GlobalGraphicsResources::lookup_texture(const std::string& name)
{
	if (m_texture_map.find(name) == m_texture_map.end())
		return nullptr;
	else
		return m_texture_map[name];
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::Texture2D* GlobalGraphicsResources::create_texture_2d(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
{
	if (m_texture_map.find(name) == m_texture_map.end())
	{
		dw::Texture2D* texture = new dw::Texture2D(w, h, array_size, mip_levels, num_samples, internal_format, format, type);
		m_texture_map[name] = texture;

		return texture;
	}
	else
	{
		DW_LOG_ERROR("A texture with the requested name (" + name + ") already exists. Returning nullptr...");
		return nullptr;
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::TextureCube* GlobalGraphicsResources::create_texture_cube(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t array_size, uint32_t mip_levels)
{
	if (m_texture_map.find(name) == m_texture_map.end())
	{
		dw::TextureCube* texture = new dw::TextureCube(w, h, array_size, mip_levels, internal_format, format, type);
		m_texture_map[name] = texture;

		return texture;
	}
	else
	{
		DW_LOG_ERROR("A texture with the requested name (" + name + ") already exists. Returning nullptr...");
		return nullptr;
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GlobalGraphicsResources::destroy_texture(const std::string& name)
{
	if (m_texture_map.find(name) != m_texture_map.end())
	{
		dw::Texture* texture = m_texture_map[name];
		DW_SAFE_DELETE(texture);
		m_texture_map.erase(name);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::Framebuffer* GlobalGraphicsResources::lookup_framebuffer(const std::string& name)
{
	if (m_framebuffer_map.find(name) == m_framebuffer_map.end())
		return nullptr;
	else
		return m_framebuffer_map[name];
}

// -----------------------------------------------------------------------------------------------------------------------------------

dw::Framebuffer* GlobalGraphicsResources::create_framebuffer(const std::string& name)
{
	if (m_framebuffer_map.find(name) == m_framebuffer_map.end())
	{
		dw::Framebuffer* fbo = new dw::Framebuffer();
		m_framebuffer_map[name] = fbo;

		return fbo;
	}
	else
	{
		DW_LOG_ERROR("A framebuffer with the requested name (" + name + ") already exists. Returning nullptr...");
		return nullptr;
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GlobalGraphicsResources::destroy_framebuffer(const std::string& name)
{
	if (m_framebuffer_map.find(name) != m_framebuffer_map.end())
	{
		dw::Framebuffer* fbo = m_framebuffer_map[name];
		DW_SAFE_DELETE(fbo);
		m_framebuffer_map.erase(name);
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------
