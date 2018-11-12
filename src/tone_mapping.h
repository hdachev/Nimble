#pragma once

#include "post_process_renderer.h"

namespace nimble
{
	enum ToneMappingOperator
	{
		TONE_MAPPING_LINEAR = 0,
		TONE_MAPPING_REINHARD = 1,
		TONE_MAPPING_HAARM_PETER_DUIKER = 2,
		TONE_MAPPING_FILMIC = 3, //Jim Hejl and Richard Burgess-Dawson
		TONE_MAPPING_UNCHARTED_2 = 4
	};

	static const char* g_tone_mapping_operators[] = 
	{ 
		"Linear",
		"Reinhard",
		"Haarm-Peter Duiker",
		"Filmic",
		"Uncharted 2"
	};

	class ToneMapping
	{
	public:
		ToneMapping();
		~ToneMapping();
		void initialize(uint16_t width, uint16_t height);
		void shutdown();
		void on_window_resized(uint16_t width, uint16_t height);
		void render(uint32_t w, uint32_t h);

		inline int32_t current_operator() { return m_current_operator; }
		inline void set_current_operator(int32_t current_operator) { m_current_operator = current_operator; }

	private:
		int32_t m_current_operator;
		float m_exposure;
		float m_uc2_exposure_bias;

		Texture* m_tone_mapped_rt;
		Framebuffer* m_tone_mapped_fbo;

		Shader* m_tone_mapping_vs;
		Shader* m_tone_mapping_fs;
		Program* m_tone_mapping_program;

		PostProcessRenderer m_post_process_renderer;
	};
}