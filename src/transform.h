#pragma once

#include <glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <gtx/quaternion.hpp>

namespace 
{
	struct Transform
	{
		glm::vec3 position;
		glm::vec3 euler;
		glm::quat orientation;
		glm::vec3 scale;
		glm::mat4 model;
		glm::mat4 prev_model;

		// -----------------------------------------------------------------------------------------------------------------------------------

		Transform()
		{
			position = glm::vec3(0.0f);
			scale = glm::vec3(1.0f);
			euler = glm::vec3(0.0f);
			orientation = glm::quat(glm::radians(euler));
			model = glm::mat4(1.0f);
			prev_model = glm::mat4(1.0f);
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline glm::vec3 forward()
		{
			return orientation * glm::vec3(0.0f, 0.0f, 1.0f);
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline glm::vec3 up()
		{
			return orientation * glm::vec3(0.0f, 1.0f, 0.0f);
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline glm::vec3 left()
		{
			return orientation * glm::vec3(1.0f, 0.0f, 0.0f);
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline void set_orientation_from_euler_yxz(const glm::vec3& e)
		{
			glm::quat pitch = glm::quat(glm::vec3(glm::radians(e.x), glm::radians(0.0f), glm::radians(0.0f)));
			glm::quat yaw = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(e.y), glm::radians(0.0f)));
			glm::quat roll = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(0.0f), glm::radians(e.z)));

			orientation = yaw * pitch * roll;
			euler = e;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline void set_orientation_from_euler_xyz(const glm::vec3& e)
		{
			glm::quat pitch = glm::quat(glm::vec3(glm::radians(e.x), glm::radians(0.0f), glm::radians(0.0f)));
			glm::quat yaw = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(e.y), glm::radians(0.0f)));
			glm::quat roll = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(0.0f), glm::radians(e.z)));

			orientation = pitch * yaw * roll;
			euler = e;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline void rotate_euler_yxz(const glm::vec3& e)
		{
			euler += e;

			glm::quat pitch = glm::quat(glm::vec3(glm::radians(euler.x), glm::radians(0.0f), glm::radians(0.0f)));
			glm::quat yaw = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(euler.y), glm::radians(0.0f)));
			glm::quat roll = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(0.0f), glm::radians(euler.z)));

			orientation = yaw * pitch * roll;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline void rotate_euler_xyz(const glm::vec3& e)
		{
			euler += e;

			glm::quat pitch = glm::quat(glm::vec3(glm::radians(euler.x), glm::radians(0.0f), glm::radians(0.0f)));
			glm::quat yaw = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(euler.y), glm::radians(0.0f)));
			glm::quat roll = glm::quat(glm::vec3(glm::radians(0.0f), glm::radians(0.0f), glm::radians(euler.z)));

			orientation = pitch * yaw * roll;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------

		inline void update()
		{
			prev_model = model;

			glm::mat4 R = glm::mat4_cast(orientation);
			glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
			glm::mat4 T = glm::translate(glm::mat4(1.0f), position);

			prev_model = model;
			model = T * R * S;
		}

		// -----------------------------------------------------------------------------------------------------------------------------------
	};
}