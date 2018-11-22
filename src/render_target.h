#pragma once

#include <memory>
#include "ogl.h"

namespace nimble
{
	struct RenderTarget
	{
		uint32_t graph_id;
		uint32_t node_id;
		uint32_t last_dependent_node_id;
		bool scaled;
		bool expired;
		float scale_w;
		float scale_h;
		uint32_t w;
		uint32_t h;
		GLenum target;
		GLenum internal_format;
		GLenum format; 
		GLenum type; 
		uint32_t num_samples;
		uint32_t array_size;
		uint32_t mip_levels;

		std::shared_ptr<Texture> texture;
	};
}
