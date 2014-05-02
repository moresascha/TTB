#version 440 core

uniform mat4 view;
uniform mat4 perspective;

in vec3 vs_in_pos;
in vec2 vs_in_tx;

out vec4 gl_Position;

void main()
{
	gl_Position = perspective * view * vec4(vs_in_pos, 1);
}