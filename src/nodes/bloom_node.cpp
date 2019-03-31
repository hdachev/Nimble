#include "bloom_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"
#include "../profiler.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(BloomNode)

// -----------------------------------------------------------------------------------------------------------------------------------

BloomNode::BloomNode(RenderGraph* graph) :
    RenderNode(graph)
{
    m_enabled   = true;
    m_threshold = 1.0f;
    m_strength  = 0.65f;
}

// -----------------------------------------------------------------------------------------------------------------------------------

BloomNode::~BloomNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BloomNode::declare_connections()
{
    register_input_render_target("Color");

    m_composite_rt = register_scaled_output_render_target("Bloom", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);

    // Clear earlier render targets.
    for (uint32_t i = 0; i < BLOOM_TEX_CHAIN_SIZE; i++)
    {
        uint32_t scale = pow(2, i);

        std::string rt_name = "Intermediate_";
        rt_name += std::to_string(scale);

        m_bloom_rt[i] = register_scaled_intermediate_render_target(rt_name, 1.0f / float(scale), 1.0f / float(scale), GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool BloomNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);
    register_float_parameter("Threshold", m_threshold, 0.0f, 1.0f);
    register_float_parameter("Strength", m_strength, 0.0f, 1.0f);

    m_color_rt      = find_input_render_target("Color");
    m_composite_rtv = RenderTargetView(0, 0, 0, m_composite_rt->texture);

    for (uint32_t i = 0; i < BLOOM_TEX_CHAIN_SIZE; i++)
        m_bloom_rtv[i] = RenderTargetView(0, 0, 0, m_bloom_rt[i]->texture);

    m_triangle_vs         = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_bright_pass_fs      = res_mgr->load_shader("shader/post_process/bloom/bright_pass_fs.glsl", GL_FRAGMENT_SHADER);
    m_bloom_downsample_fs = res_mgr->load_shader("shader/post_process/bloom/downsample_fs.glsl", GL_FRAGMENT_SHADER);
    m_bloom_upsample_fs   = res_mgr->load_shader("shader/post_process/bloom/upsample_fs.glsl", GL_FRAGMENT_SHADER);
    m_bloom_composite_fs  = res_mgr->load_shader("shader/post_process/bloom/composite_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_triangle_vs)
    {
        if (m_bright_pass_fs)
            m_bright_pass_program = renderer->create_program(m_triangle_vs, m_bright_pass_fs);
        else
            return false;

        if (m_bloom_downsample_fs)
            m_bloom_downsample_program = renderer->create_program(m_triangle_vs, m_bloom_downsample_fs);
        else
            return false;

        if (m_bloom_upsample_fs)
            m_bloom_upsample_program = renderer->create_program(m_triangle_vs, m_bloom_upsample_fs);
        else
            return false;

        if (m_bloom_composite_fs)
            m_bloom_composite_program = renderer->create_program(m_triangle_vs, m_bloom_composite_fs);
        else
            return false;

        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BloomNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    blit_render_target(renderer, m_color_rt, m_composite_rt);

    if (m_enabled)
    {
        bright_pass(renderer);
        downsample(renderer);
        upsample(renderer);
        composite(renderer);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BloomNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string BloomNode::name()
{
    return "Bloom";
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BloomNode::bright_pass(Renderer* renderer)
{
    NIMBLE_SCOPED_SAMPLE("Bright Pass");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Bright Pass");

    m_bright_pass_program->use();

    if (m_bright_pass_program->set_uniform("s_Color", 0))
        m_color_rt->texture->bind(0);

    m_bright_pass_program->set_uniform("u_Threshold", m_threshold);

    renderer->bind_render_targets(1, &m_bloom_rtv[0], nullptr);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());
    glClear(GL_COLOR_BUFFER_BIT);

    render_fullscreen_triangle(renderer, nullptr);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BloomNode::downsample(Renderer* renderer)
{
    NIMBLE_SCOPED_SAMPLE("Downsample");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Downsample");

    m_bloom_downsample_program->use();

    // Progressively blur bright pass into blur textures.
    for (uint32_t i = 0; i < (BLOOM_TEX_CHAIN_SIZE - 1); i++)
    {
        float scale = pow(2, i + 1);

        glm::vec2 pixel_size = glm::vec2(1.0f / (float(m_graph->window_width()) / scale), 1.0f / (float(m_graph->window_height()) / scale));
        m_bloom_downsample_program->set_uniform("u_PixelSize", pixel_size);

        if (m_bloom_downsample_program->set_uniform("s_Texture", 0))
            m_bloom_rt[i]->texture->bind(0);

        renderer->bind_render_targets(1, &m_bloom_rtv[i + 1], nullptr);
        glViewport(0, 0, m_graph->window_width() / scale, m_graph->window_height() / scale);
        glClear(GL_COLOR_BUFFER_BIT);

        render_fullscreen_triangle(renderer, nullptr);
    }

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

// TODO: Prevent clearing when upsampling and use additive blending.
void BloomNode::upsample(Renderer* renderer)
{
    NIMBLE_SCOPED_SAMPLE("Upsample");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Upsample");

    m_bloom_upsample_program->use();

    m_bloom_upsample_program->set_uniform("u_Strength", m_enabled ? m_strength : 0.0f);

#ifdef BLOOM_ADDITIVE_BLEND
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
#endif

    // Upsample each downsampled target
    for (uint32_t i = 0; i < (BLOOM_TEX_CHAIN_SIZE - 1); i++)
    {
        float scale = pow(2, BLOOM_TEX_CHAIN_SIZE - i - 2);

        glm::vec2 pixel_size = glm::vec2(1.0f / (float(m_graph->window_width()) / scale), 1.0f / (float(m_graph->window_height()) / scale));
        m_bloom_upsample_program->set_uniform("u_PixelSize", pixel_size);

        if (m_bloom_upsample_program->set_uniform("s_Texture", 0))
            m_bloom_rt[BLOOM_TEX_CHAIN_SIZE - i - 1]->texture->bind(0);

        renderer->bind_render_targets(1, &m_bloom_rtv[BLOOM_TEX_CHAIN_SIZE - i - 2], nullptr);
        glViewport(0, 0, m_graph->window_width() / scale, m_graph->window_height() / scale);

#ifndef BLOOM_ADDITIVE_BLEND
		glClear(GL_COLOR_BUFFER_BIT);
#endif
        render_fullscreen_triangle(renderer, nullptr);
    }

#ifdef BLOOM_ADDITIVE_BLEND
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
#endif

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void BloomNode::composite(Renderer* renderer)
{
    NIMBLE_SCOPED_SAMPLE("Composite");

    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Composite");

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    m_bloom_composite_program->use();

    m_bloom_composite_program->set_uniform("u_Strength", m_enabled ? m_strength : 0.0f);

    if (m_bloom_composite_program->set_uniform("s_Bloom", 0))
        m_bloom_rt[0]->texture->bind(0);

    renderer->bind_render_targets(1, &m_composite_rtv, nullptr);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    render_fullscreen_triangle(renderer, nullptr);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble