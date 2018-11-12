#include "hi_z_buffer.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "logger.h"
#include "gpu_profiler.h"
#include "imgui.h"

namespace nimble
{
	HiZBuffer::HiZBuffer()
	{

	}

	HiZBuffer::~HiZBuffer()
	{

	}

	void HiZBuffer::initialize(uint16_t width, uint16_t height)
	{
		on_window_resized(width, height);

		std::string vs_path = "shader/post_process/quad_vs.glsl";
		m_quad_vs = GlobalGraphicsResources::load_shader(GL_VERTEX_SHADER, vs_path);

		{
			std::string fs_path = "shader/post_process/hiz/hiz_fs.glsl";
			m_hiz_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_quad_vs, m_hiz_fs };
			std::string combined_path = vs_path + fs_path;
			m_hiz_program = GlobalGraphicsResources::load_program(combined_path, 2, &shaders[0]);

			if (!m_quad_vs || !m_hiz_fs || !m_hiz_program)
			{
				NIMBLE_LOG_ERROR("Failed to load Hi-Z shaders");
			}
		}

		// Copy
		{
			std::string fs_path = "shader/post_process/hiz/copy_fs.glsl";
			m_copy_fs = GlobalGraphicsResources::load_shader(GL_FRAGMENT_SHADER, fs_path);

			Shader* shaders[] = { m_quad_vs, m_copy_fs };
			std::string combined_name = vs_path + fs_path;
			m_copy_program = GlobalGraphicsResources::load_program(combined_name, 2, &shaders[0]);

			if (!m_quad_vs || !m_copy_fs || !m_copy_program)
			{
				NIMBLE_LOG_ERROR("Failed to load Copy shaders");
			}
		}
	}

	void HiZBuffer::shutdown()
	{

	}

	void HiZBuffer::profiling_gui()
	{
		ImGui::Text("HiZ - Copy: %f ms", GPUProfiler::result("HiZ - Copy"));
		ImGui::Text("HiZ - Downsample: %f ms", GPUProfiler::result("HiZ - Downsample"));
	}

	void HiZBuffer::on_window_resized(uint16_t width, uint16_t height)
	{
		uint32_t w = width;
		uint32_t h = height;
		uint32_t count = 0;

		while (w > 8 && h > 8)
		{
			count++;
			w /= 2;
			h /= 2;
		}

		for (auto fbo : m_fbos)
			delete fbo;

		m_fbos.clear();
		
		GlobalGraphicsResources::destroy_texture(RENDER_TARGET_HiZ);

		m_hiz_rt = GlobalGraphicsResources::create_texture_2d(RENDER_TARGET_HiZ, width, height, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, 1, count);
		m_hiz_rt->generate_mipmaps();
		m_hiz_rt->set_min_filter(GL_LINEAR);
		m_hiz_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		for (int i = 0; i <= count; i++)
		{
			Framebuffer* fbo = new Framebuffer();
			fbo->attach_depth_stencil_target(m_hiz_rt, 0, i);

			m_fbos.push_back(fbo);
		}
	}

	void HiZBuffer::render(uint32_t w, uint32_t h)
	{
		// Copy G-Buffer Depth into Mip 0 of HiZ
		copy_depth(w, h);

		// Generate HiZ Chain
		downsample(w, h);
	}

	void HiZBuffer::copy_depth(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("HiZ - Copy");

		m_copy_program->use();

		if (m_copy_program->set_uniform("s_Texture", 0))
			GlobalGraphicsResources::lookup_texture(RENDER_TARGET_GBUFFER_DEPTH)->bind(0);

		m_post_process_renderer.render(w, h, m_fbos[0], GL_DEPTH_BUFFER_BIT);

		GPUProfiler::end("HiZ - Copy");
	}

	void HiZBuffer::downsample(uint32_t w, uint32_t h)
	{
		GPUProfiler::begin("HiZ - Downsample");

		m_hiz_program->use();

		for (uint32_t i = 1; i < m_fbos.size(); i++)
		{
			float scale = pow(2, i);

			if (m_hiz_program->set_uniform("s_Texture", 0))
				m_hiz_rt->bind(0);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, i - 1);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, i - 1);

			m_post_process_renderer.render(w / scale, h / scale, m_fbos[i], GL_DEPTH_BUFFER_BIT);
		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);

		GPUProfiler::end("HiZ - Downsample");
	}
}