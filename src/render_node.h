#pragma once

#include <functional>

#include "render_target.h"
#include "global_graphics_resources.h"
#include "view.h"

namespace nimble
{
	struct View;
	struct FramebufferGroup;
	class RenderGraph;

	enum RenderNodeType
	{
		RENDER_NODE_SCENE = 0,
		RENDER_NODE_FULLSCREEN = 1,
		RENDER_NODE_COMPUTE = 2
	};

	enum RenderNodeFlags
	{
		NODE_USAGE_PER_OBJECT_UBO = BIT_FLAG(0),
		NODE_USAGE_PER_VIEW_UBO = BIT_FLAG(1),
		NODE_USAGE_STATIC_MESH = BIT_FLAG(2),
		NODE_USAGE_SKELETAL_MESH = BIT_FLAG(3),
		NODE_USAGE_MATERIAL_ALBEDO = BIT_FLAG(4),
		NODE_USAGE_MATERIAL_NORMAL = BIT_FLAG(5),
		NODE_USAGE_MATERIAL_METAL_SPEC = BIT_FLAG(6),
		NODE_USAGE_MATERIAL_ROUGH_SMOOTH = BIT_FLAG(7),
		NODE_USAGE_MATERIAL_DISPLACEMENT = BIT_FLAG(8),
		NODE_USAGE_MATERIAL_EMISSIVE = BIT_FLAG(9),
		NODE_USAGE_ALL_MATERIALS = NODE_USAGE_MATERIAL_ALBEDO | NODE_USAGE_MATERIAL_NORMAL | NODE_USAGE_MATERIAL_METAL_SPEC | NODE_USAGE_MATERIAL_ROUGH_SMOOTH | NODE_USAGE_MATERIAL_EMISSIVE | NODE_USAGE_MATERIAL_DISPLACEMENT
	};

	struct SceneRenderDesc
	{
		FramebufferGroup* fbg;
		uint32_t target_slice;
		uint32_t x;
		uint32_t y;
		uint32_t w;
		uint32_t h;
		GLenum clear_flags = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
		uint32_t num_clear_colors = 0;
		float clear_colors[8][4];
		double clear_depth = 1;
	};

	class RenderNode
	{
	public:
		RenderNode(RenderNodeType type, RenderGraph* graph);
		~RenderNode();

		RenderTarget* render_target_by_name(const std::string& name);
		RenderTarget* render_target_dependecy_by_name(const std::string& name);
		Buffer* buffer_dependecy_by_name(const std::string& name);
		void set_dependency(const std::string& name, RenderTarget* rt);
		void set_dependency(const std::string& name, Buffer* buffer);
		void timing_total(float& cpu_time, float& gpu_time);

		// Inline getters
		inline RenderNodeType type() { return m_render_node_type; }
		inline bool is_enabled() { return m_enabled; }
		inline uint32_t id() { return m_id; }

		// Inline setters
		inline void enable() { m_enabled = true; }
		inline void disable() { m_enabled = false; }

		// Virtual methods
		virtual void passthrough();
		virtual void execute(const View& view) = 0;
		virtual bool initialize() = 0;
		virtual void shutdown() = 0;
		virtual uint32_t flags() = 0;
		virtual std::string name() = 0;

	protected:
		std::shared_ptr<RenderTarget> register_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);
		std::shared_ptr<RenderTarget> register_scaled_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples = 1, uint32_t array_size = 1, uint32_t mip_levels = 1);

	protected:
		RenderGraph* m_graph;
		float m_total_time_cpu;
		float m_total_time_gpu;
		std::string m_passthrough_name;
		
	private:
		bool m_enabled;
		uint32_t m_id;
		RenderNodeType m_render_node_type;
		std::unordered_map<std::string, RenderTarget*> m_render_targets;
		std::unordered_map<std::string, RenderTarget*> m_rt_dependecies;
		std::unordered_map<std::string, Buffer*> m_buffer_dependecies;
	};

	class Scene;
	class ShaderLibrary;

	class SceneRenderNode : public RenderNode
	{
	public:
		struct Params
		{
			Scene* scene;
			ShaderLibrary* library;
			View* view;
			uint32_t num_rt_views;
			RenderTargetView* rt_views;
			RenderTargetView* depth_views;
			uint32_t x;
			uint32_t y;
			uint32_t w;
			uint32_t h;
			GLenum clear_flags = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
			uint32_t num_clear_colors = 0;
			float clear_colors[8][4];
			double clear_depth = 1;
		};

		SceneRenderNode(RenderGraph* graph);
		~SceneRenderNode();

		void execute(const View& view) override;

	protected:
		void render_scene(const Params& params);
	};

	class MultiPassRenderNode : public RenderNode
	{
	public:
		MultiPassRenderNode(RenderNodeType type, RenderGraph* graph);
		~MultiPassRenderNode();

		void execute(const View& view) override;
		void timing_sub_pass(const uint32_t& index, std::string& name, float& cpu_time, float& gpu_time);

		// Inline getters
		inline uint32_t sub_pass_count() { return m_sub_passes.size(); }

	protected:
		void attach_sub_pass(const std::string& node_name, std::function<void(void)> function);

	private:
		std::vector<std::pair<std::string, std::function<void(void)>>> m_sub_passes;
		std::vector<std::pair<float, float>> m_sub_pass_timings;
	};

	class FullscreenRenderNode : public MultiPassRenderNode
	{
	public:
		struct Params
		{
			Scene* scene;
			View* view;
			uint32_t num_rt_views;
			RenderTargetView* rt_views;
			uint32_t x;
			uint32_t y;
			uint32_t w;
			uint32_t h;
			GLenum clear_flags = GL_COLOR_BUFFER_BIT;
			uint32_t num_clear_colors = 0;
			float clear_colors[8][4];
		};

		FullscreenRenderNode(RenderGraph* graph);
		~FullscreenRenderNode();

	protected:
		void render_triangle(const Params& params);
	};

	class ComputeRenderNode : public MultiPassRenderNode
	{
	public:
		ComputeRenderNode(RenderGraph* graph);
		~ComputeRenderNode();
	};
}