#pragma once

#include <memory>
#include "ogl.h"

namespace nimble
{
	struct RenderTarget
	{
		uint32_t id;
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

		RenderTarget();
	};

	struct RenderTargetView
	{
		uint32_t face;
		uint32_t layer;
		uint32_t mip_level;
		RenderTarget* render_target;

		RenderTargetView()
		{
			face = 0;
			layer = 0;
			mip_level = 0;
			render_target = nullptr;
		}

		RenderTargetView(uint32_t _face, uint32_t _layer, uint32_t _mip_level, RenderTarget* _render_target)
		{
			face = _render_target->target == GL_TEXTURE_CUBE_MAP ? _face : 0;
			layer = _layer;
			mip_level = _mip_level;
			render_target = _render_target;
		}
	};
}
