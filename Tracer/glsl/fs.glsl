#version 150

uniform samplerBuffer sampler;
uniform int width;
in vec2 fs_in_tc;

out vec4 fs_out_color;

layout (pixel_center_integer) in vec4 gl_FragCoord;

void main()
{
	int index = int(width * gl_FragCoord.y + gl_FragCoord.x);
	fs_out_color = texelFetch(sampler, index);
}