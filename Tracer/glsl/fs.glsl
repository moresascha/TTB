#version 150

uniform samplerBuffer sampler;
uniform int width;
uniform int height;

uniform int twidth;
uniform int theight;

in vec2 fs_in_tc;

out vec4 fs_out_color;

layout (pixel_center_integer) in vec4 gl_FragCoord;

void main()
{
    int stridex = int(twidth / float(width) + 0.5);
    int stridey = int(theight / float(height) + 0.5);
    stridex = stridex < 1 ? 1 : stridex;
    stridey = stridey < 1 ? 1 : stridey;
	int index = int(twidth * stridey * gl_FragCoord.y + stridex * gl_FragCoord.x);
    //int index = int(width * (height * (1-fs_in_tc.y)) + width * fs_in_tc.x);
    fs_out_color = texelFetch(sampler, index);
}