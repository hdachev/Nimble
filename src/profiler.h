#pragma once

#include "ogl.h"
#include "timer.h"
#include <memory>
#include <unordered_map>
#include <string>

namespace nimble
{
#define NUM_BUFFERED_QUERIES 3

struct ProfileScope
{
    Query    queries[NUM_BUFFERED_QUERIES];
    uint32_t index = 0;
    Timer    timer;
    float    last_result_gpu = 0;
    float    last_result_cpu = 0;
};

struct CPUProfileScope
{
    uint32_t index = 0;
    Timer    timer;
    float    last_result_cpu = 0;
};

class Profiler
{
public:
    static void shutdown();
    static void begin_sample(std::string name);
    static void end_sample(std::string name);
    static void begin_cpu_sample(std::string name);
    static void end_cpu_sample(std::string name);
    static void result(std::string name, float& cpu_sample, float& gpu_sample);
    static void cpu_result(std::string name, float& cpu_sample);

private:
    static std::unordered_map<std::string, std::unique_ptr<ProfileScope>>    m_scopes;
    static std::unordered_map<std::string, std::unique_ptr<CPUProfileScope>> m_cpu_scopes;
};
} // namespace nimble