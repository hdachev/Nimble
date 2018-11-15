#pragma once

#include <string>
#include <stdint.h>
#include <glm.hpp>
#include "macros.h"
#include "murmur_hash.h"

namespace nimble
{
	class Texture;
	class Program;
	class UniformBuffer;

	#define MAX_MATERIAL_TEXTURES 8

	struct MaterialUniforms
	{
		glm::vec4 albedo;
		glm::vec4 emissive;
		glm::vec4 metal_rough;
	};

	enum ShadingModel
	{
		SHADING_MODEL_STANDARD = 0,
		SHADING_MODEL_CLOTH,
		SHADING_MODEL_SUBSURFACE
	};

	enum LightingModel
	{
		LIGHTING_MODEL_LIT = 0,
		LIGHTING_MODEL_UNLIT
	};

	enum DisplacementType
	{
		DISPLACEMENT_NONE = 0,
		DISPLACEMENT_PARALLAX_OCCLUSION,
		DISPLACEMENT_TESSELLATION
	};

	enum BlendMode
	{
		BLEND_MODE_OPAQUE = 0,
		BLEND_MODE_ADDITIVE,
		BLEND_MODE_MASKED,
		BLEND_MODE_TRANSLUCENT
	};

	class Material
	{
		friend class ResourceManager;

	public:
		Material();
		~Material();

		// Bind textures and material constants to pipeline.
		void bind(Program* program, UniformBuffer* ubo);

		// Property setters.
		inline void set_name(const std::string& name) { m_name = name; m_hash = NIMBLE_HASH(m_name.c_str()); }
		inline void set_metallic_workflow(bool metallic) { m_metallic_workflow = metallic; }
		inline void set_double_sided(bool double_sided) { m_double_sided = double_sided; }
		inline void set_uniform_albedo(const glm::vec4& albedo) { m_uniforms.albedo = albedo; }
		inline void set_uniform_emissive(const glm::vec3& emissive) { m_uniforms.emissive = glm::vec4(emissive, 0.0f); }
		inline void set_uniform_metallic(const float& metallic) { m_uniforms.metal_rough.x = metallic; }
		inline void set_uniform_roughness(const float& roughness) { m_uniforms.metal_rough.y = roughness; }
		inline void set_vertex_shader_func(const std::string& src) { m_vertex_shader_func = src; m_has_vertex_shader_func = true; }
		inline void set_fragment_shader_func(const std::string& src) { m_fragment_shader_func = src; m_has_fragment_shader_func = true; }

		// Property getters
		inline std::string name() { return m_name; }
		inline uint64_t hash() { return m_hash; }
		inline bool has_vertex_shader_func() { return m_has_vertex_shader_func; }
		inline bool has_fragment_shader_func() { return m_has_fragment_shader_func; }
		inline std::string vertex_shader_func() { return m_vertex_shader_func; }
		inline std::string fragment_shader_func() { return m_fragment_shader_func; }
		inline bool is_metallic_workflow() { return m_metallic_workflow; }
		inline bool is_double_sided() { return m_double_sided; }
		inline glm::vec4 uniform_albedo() { return m_uniforms.albedo; }
		inline glm::vec3 uniform_emissive() { return glm::vec3(m_uniforms.emissive); }
		inline float uniform_metallic() { return m_uniforms.metal_rough.x; }
		inline float uniform_roughness() { return m_uniforms.metal_rough.y; }
		inline MaterialUniforms uniforms() { return m_uniforms; }

	private:
		std::string					  m_name;
		uint64_t					  m_hash;
		bool						  m_metallic_workflow;
		bool						  m_double_sided;
		std::string					  m_vertex_shader_func;
		bool						  m_has_vertex_shader_func = false;
		std::string					  m_fragment_shader_func;
		bool						  m_has_fragment_shader_func = false;
		BlendMode					  m_blend_mode;
		DisplacementType			  m_displacement_type;
		ShadingModel				  m_shading_model;
		LightingModel				  m_lighting_model;
		uint32_t					  m_texture_count;
		std::pair<Texture*, uint32_t> m_textures[MAX_MATERIAL_TEXTURES];
		MaterialUniforms			  m_uniforms;
		bool						  m_has_uniforms;
	};
}