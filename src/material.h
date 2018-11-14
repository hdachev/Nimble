#pragma once

#include <string>
#include <stdint.h>
#include <glm.hpp>
#include "macros.h"

namespace nimble
{
	#define MAX_MATERIAL_TEXTURES 8

	struct MaterialUniforms
	{
		glm::vec4 albedo;
		glm::vec4 emissive;
		glm::vec4 specular;
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

	private:
		std::string					  m_name;
		uint64_t					  m_hash;
		bool						  m_metallic_workflow;
		bool						  m_double_sided;
		std::string					  m_vertex_shader_func;
		std::string					  m_fragment_shader_func;
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