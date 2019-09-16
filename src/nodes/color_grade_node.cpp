#include "color_grade_node.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ColorGradeNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ColorGradeNode::ColorGradeNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ColorGradeNode::~ColorGradeNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ColorGradeNode::declare_connections()
{
    register_input_render_target("Color");

    m_output_rt = register_scaled_output_render_target("ColorGrade", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ColorGradeNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_texture = find_input_render_target("Color");

    m_output_rtv = RenderTargetView(0, 0, 0, m_output_rt->texture);

	int w, y, c;

	struct LUTElement
	{
		stbi_uc r;
		stbi_uc g;
		stbi_uc b;
	};

    LUTElement* data = (LUTElement*)stbi_load("RGBTable16x1.jpg", &w, &y, &c, 3);

    std::vector<stbi_uc> lut_data;
    int                  i = 0;

    lut_data.resize(16 * 16 * 16 * 3);

    for (int z = 0; z < 16; z++)
    {
        for (int y = 0; y < 16; y++)
        {
			for (int x = 0; x < 16; x++)
			{
				lut_data[i++] = data[z * 16 + x + (16 * 16 * y)].r;
				lut_data[i++] = data[z * 16 + x + (16 * 16 * y)].g;
				lut_data[i++] = data[z * 16 + x + (16 * 16 * y)].b;
			}
        }
    }

    m_lut = std::make_unique<Texture3D>(16, 16, 16, 1, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
    m_lut->set_data(0, lut_data.data());

	m_lut->set_min_filter(GL_LINEAR);
    m_lut->set_mag_filter(GL_LINEAR);
    m_lut->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

	stbi_image_free(data);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/post_process/color_grade/color_grade_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_vs && m_fs)
    {
        m_program = renderer->create_program(m_vs, m_fs);

        if (m_program)
            return true;
        else
        {
            NIMBLE_LOG_ERROR("Failed to create Program!");
            return false;
        }
    }
    else
    {
        NIMBLE_LOG_ERROR("Failed to load Shaders!");
        return false;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ColorGradeNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_program->use();

    renderer->bind_render_targets(1, &m_output_rtv, nullptr);

    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    if (m_program->set_uniform("s_Texture", 0) && m_texture)
        m_texture->texture->bind(0);

    if (m_program->set_uniform("s_LUT", 1))
        m_lut->bind(1);

    render_fullscreen_triangle(renderer, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ColorGradeNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ColorGradeNode::name()
{
    return "Color Grade";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble