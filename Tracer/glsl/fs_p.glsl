#version 440 core

uniform vec3 color;

out vec4 fs_out_color;

void main()
{
	fs_out_color = vec4(color, 1);
}