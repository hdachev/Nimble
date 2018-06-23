#pragma once

#include <unordered_map>
#include <ogl.h>
#include <string>

// Class for associating names with graphics resources and offering a global point-of-access for all render passes.
class GlobalGraphicsResources
{
public:
	static void initialize();
	// Cleanup all allocated resources.
	static void shutdown();

	// Lookup a previously created texture by name.
	static dw::Texture* lookup_texture(const std::string& name);

	// Texture create methods. Returns nullptr if a texture with the same name already exists,
	static dw::Texture2D* create_texture_2d(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
	static dw::TextureCube* create_texture_cube(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum internal_format, GLenum format, GLenum type, uint32_t array_size = 1, uint32_t mip_levels = 1);

	// Texture destroy method.
	static void destroy_texture(const std::string& name);

	// Lookup a previously created framebuffer by name.
	static dw::Framebuffer* lookup_framebuffer(const std::string& name);

	// Framebuffer create method. Returns nullptr if a framebuffer with the same name already exists,
	static dw::Framebuffer* create_framebuffer(const std::string& name);

	// Framebuffer destroy method.
	static void destroy_framebuffer(const std::string& name);

	// Uniform buffer getters.
	inline static dw::UniformBuffer* per_frame_ubo() { return m_per_frame; }
	inline static dw::UniformBuffer* per_scene_ubo() { return m_per_scene; }
	inline static dw::UniformBuffer* per_entity_ubo() { return m_per_entity; }

private:
	// Resource maps.
	static std::unordered_map<std::string, dw::Texture*> m_texture_map;
	static std::unordered_map<std::string, dw::Framebuffer*> m_framebuffer_map;

	// Uniform buffers.
	static dw::UniformBuffer* m_per_frame;
	static dw::UniformBuffer* m_per_scene;
	static dw::UniformBuffer* m_per_entity;
};