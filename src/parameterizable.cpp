#include "parameterizable.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

Parameterizable::Parameterizable()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

Parameterizable::~Parameterizable()
{

}

// -----------------------------------------------------------------------------------------------------------------------------------

void Parameterizable::set_bool_parameter(const std::string& name, bool value)
{
    for (auto& param : m_bool_parameters)
    {
        if (name == param.name && param.ptr)
        {
            *param.ptr = value;
            return;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Parameterizable::set_int_parameter(const std::string& name, int32_t value)
{
    for (auto& param : m_int_parameters)
    {
        if (name == param.name && param.ptr)
        {
            *param.ptr = value;
            return;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Parameterizable::set_float_parameter(const std::string& name, float value)
{
    for (auto& param : m_float_parameters)
    {
        if (name == param.name && param.ptr)
        {
            *param.ptr = value;
            return;
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

BoolParameter* Parameterizable::bool_parameters(int32_t& count)
{
    count = m_bool_parameters.size();

    if (count == 0)
        return nullptr;

    return &m_bool_parameters[0];
}

// -----------------------------------------------------------------------------------------------------------------------------------

IntParameter* Parameterizable::int_parameters(int32_t& count)
{
    count = m_int_parameters.size();

    if (count == 0)
        return nullptr;

    return &m_int_parameters[0];
}

// -----------------------------------------------------------------------------------------------------------------------------------

FloatParameter* Parameterizable::float_parameters(int32_t& count)
{
    count = m_float_parameters.size();

    if (count == 0)
        return nullptr;

    return &m_float_parameters[0];
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Parameterizable::register_bool_parameter(const std::string& name, bool& parameter)
{
    m_bool_parameters.push_back({ &parameter, name });
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Parameterizable::register_int_parameter(const std::string& name, int32_t& parameter, int32_t min, int32_t max)
{
    m_int_parameters.push_back({ &parameter, name, min, max });
}

// -----------------------------------------------------------------------------------------------------------------------------------

void Parameterizable::register_float_parameter(const std::string& name, float& parameter, float min, float max)
{
    m_float_parameters.push_back({ &parameter, name, min, max });
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble