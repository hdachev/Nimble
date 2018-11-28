#pragma once

#include <stdint.h>

namespace nimble
{
#define BIT_MASK(n) ((1 << n) - 1)
#define WRITE_BIT_RANGE_64(value, dst, offset, num_bits) (dst |= (static_cast<uint64_t>(value & BIT_MASK(num_bits)) << offset))
#define READ_BIT_RANGE_64(src, offset, num_bits) ((src >> offset) & BIT_MASK(num_bits))

	struct VertexShaderKey
	{
		uint64_t key = 0;

		inline void set_vertex_func_id(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 0, 10); }
		inline void set_mesh_type(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 10, 3); }
		inline void set_normal_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 13, 1); }

		inline uint32_t vertex_func_id() { return READ_BIT_RANGE_64(key, 0, 10); }
		inline uint32_t mesh_type() { return READ_BIT_RANGE_64(key, 10, 3); }
		inline uint32_t normal_texture() { return READ_BIT_RANGE_64(key, 13, 1); }
	};

	struct FragmentShaderKey
	{
		uint64_t key = 0;

		inline void set_fragment_func_id(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 0, 10); }
		inline void set_displacement_type(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 10, 2); }
		inline void set_alpha_cutout(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 12, 1); }
		inline void set_lighting_model(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 13, 1); }
		inline void set_shading_model(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 14, 2); }
		inline void set_albedo_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 16, 1); }
		inline void set_normal_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 17, 1); }
		inline void set_roughness_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 18, 1); }
		inline void set_metallic_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 19, 1); }
		inline void set_emissive_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 20, 1); }
		inline void set_metallic_workflow(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 21, 1); }
		inline void set_custom_texture_count(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 22, 3); }

		inline uint32_t fragment_func_id() { return READ_BIT_RANGE_64(key, 0, 10); }
		inline uint32_t displacement_type() { return READ_BIT_RANGE_64(key, 10, 2); }
		inline uint32_t alpha_cutout() { return READ_BIT_RANGE_64(key, 12, 1); }
		inline uint32_t lighting_model() { return READ_BIT_RANGE_64(key, 13, 1); }
		inline uint32_t shading_model() { return READ_BIT_RANGE_64(key, 14, 2); }
		inline uint32_t albedo_texture() { return READ_BIT_RANGE_64(key, 16, 1); }
		inline uint32_t normal_texture() { return READ_BIT_RANGE_64(key, 17, 1); }
		inline uint32_t roughness_texture() { return READ_BIT_RANGE_64(key, 18, 1); }
		inline uint32_t metallic_texture() { return READ_BIT_RANGE_64(key, 19, 1); }
		inline uint32_t emissive_texture() { return READ_BIT_RANGE_64(key, 20, 1); }
		inline uint32_t metallic_workflow() { return READ_BIT_RANGE_64(key, 21, 1); }
		inline uint32_t custom_texture_count() { return READ_BIT_RANGE_64(key, 22, 3); }
	};

	struct ProgramKey
	{
		uint64_t key = 0;

		ProgramKey()
		{

		}

		ProgramKey(VertexShaderKey& vs_key, FragmentShaderKey& fs_key)
		{
			set_vertex_func_id(vs_key.vertex_func_id());
			set_fragment_func_id(fs_key.fragment_func_id());
			set_mesh_type(vs_key.mesh_type());
			set_displacement_type(fs_key.displacement_type());
			set_alpha_cutout(fs_key.alpha_cutout());
			set_lighting_model(fs_key.lighting_model());
			set_shading_model(fs_key.shading_model());
			set_albedo_texture(fs_key.albedo_texture());
			set_normal_texture(fs_key.normal_texture());
			set_roughness_texture(fs_key.roughness_texture());
			set_metallic_texture(fs_key.metallic_texture());
			set_emissive_texture(fs_key.emissive_texture());
			set_metallic_workflow(fs_key.metallic_workflow());
			set_custom_texture_count(fs_key.custom_texture_count());
		}

		inline void set_vertex_func_id(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 0, 10); }
		inline void set_fragment_func_id(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 10, 10); }
		inline void set_mesh_type(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 20, 3); }
		inline void set_displacement_type(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 23, 2); }
		inline void set_alpha_cutout(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 24, 1); }
		inline void set_lighting_model(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 25, 1); }
		inline void set_shading_model(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 26, 2); }
		inline void set_albedo_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 28, 1); }
		inline void set_normal_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 29, 1); }
		inline void set_roughness_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 30, 1); }
		inline void set_metallic_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 31, 1); }
		inline void set_emissive_texture(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 32, 1); }
		inline void set_metallic_workflow(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 33, 1); }
		inline void set_custom_texture_count(const uint32_t& value) { WRITE_BIT_RANGE_64(value, key, 34, 3); }

		inline uint32_t vertex_func_id() { return READ_BIT_RANGE_64(key, 0, 10); }
		inline uint32_t fragment_func_id() { return READ_BIT_RANGE_64(key, 10, 10); }
		inline uint32_t mesh_type() { return READ_BIT_RANGE_64(key, 20, 3); }
		inline uint32_t displacement_type() { return READ_BIT_RANGE_64(key, 23, 2); }
		inline uint32_t alpha_cutout() { return READ_BIT_RANGE_64(key, 24, 1); }
		inline uint32_t lighting_model() { return READ_BIT_RANGE_64(key, 25, 1); }
		inline uint32_t shading_model() { return READ_BIT_RANGE_64(key, 26, 2); }
		inline uint32_t albedo_texture() { return READ_BIT_RANGE_64(key, 28, 1); }
		inline uint32_t normal_texture() { return READ_BIT_RANGE_64(key, 29, 1); }
		inline uint32_t roughness_texture() { return READ_BIT_RANGE_64(key, 30, 1); }
		inline uint32_t metallic_texture() { return READ_BIT_RANGE_64(key, 31, 1); }
		inline uint32_t emissive_texture() { return READ_BIT_RANGE_64(key, 32, 1); }
		inline uint32_t metallic_workflow() { return READ_BIT_RANGE_64(key, 33, 1); }
		inline uint32_t custom_texture_count() { return READ_BIT_RANGE_64(key, 34, 3); }
	};
}
