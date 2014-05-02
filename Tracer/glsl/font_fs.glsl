#version 150

uniform sampler2D tex;
in vec2 fs_in_tc;

out vec4 fs_out_color;

void main()
{
    vec4 c = texture(tex, fs_in_tc);
    fs_out_color = vec4(1,1,0,c.x);
}