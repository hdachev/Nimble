#include <iostream>
#include <fstream>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <application.h>
#include <camera.h>
#include <utility.h>
#include <material.h>
#include <macros.h>
#include <memory>
#include <debug_draw.h>
#include <imgui_helpers.h>

#include "renderer.h"
#include "scene.h"

class GraphicsDemo : public dw::Application
{
private:

protected:
	// -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {

    }

	// -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
		
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {

    }

	// -----------------------------------------------------------------------------------------------------------------------------------
};

DW_DECLARE_MAIN(GraphicsDemo)
