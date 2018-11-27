#pragma once

#include "ogl.h"

namespace nimble
{
	// 16-bits  | 3-bits	| 2-bits	   | 1-bit		  | 
	// Material | Mesh Type | Displacement | Alpha Cutout |

	struct ShaderKey
	{
		uint64_t key;

		inline void set_material_id(const uint32_t& value) { uint64_t temp = value; key |= temp; }
		inline void set_mesh_type(const uint32_t& value) { uint64_t temp = value; key |= (temp << 16); }
		inline void set_displacement_type(const uint32_t& value) { uint64_t temp = value; key |= (temp << 19); }
		inline void set_alpha_cutout(const uint32_t& value) { uint64_t temp = value; key |= (temp << 20); }

		inline uint32_t material_id() { return key & 0xffff; }
		inline uint32_t mesh_type() { return (key >> 16) & 7; }
		inline uint32_t displacement_type() { return (key >> 19) & 4; }
		inline uint32_t alpha_cutout() { return (key >> 20) & 1; }
		/*inline void SetRenderPass(int _renderPass) { uint64 temp = _renderPass; key |= temp << 59; }
		inline void SetRenderSubpass(int _renderSubPass) { uint64 temp = _renderSubPass; key |= temp << 55; }
		inline void SetMaterial(int _material) { uint64 temp = _material; key |= temp << 45; }
		inline void SetVertexArray(int _vertexArray) { uint64 temp = _vertexArray; key |= temp << 35; }
		inline void SetDepthTest(int _depthTest) { uint64 temp = _depthTest; key |= temp << 34; }
		inline void SetStencilTest(int _stencilTest) { uint64 temp = _stencilTest; key |= temp << 33; }
		inline void SetFaceCulling(int _faceCull) { uint64 temp = _faceCull; key |= temp << 31; }
		inline void SetDepth(float _depth) { key |= (static_cast<uint64>(_depth) & 0xffff); }*/
	};

	class ShaderLibrary
	{
	public:

	};
}