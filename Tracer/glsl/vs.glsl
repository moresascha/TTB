#version 150

out vec4 gl_Position;
out vec2 fs_in_tc;

void main()
{
	vec2 pos;
	pos.x = (gl_VertexID / 2) * 4.0 - 1.0;
	pos.y = (gl_VertexID % 2) * 4.0 - 1.0;
	gl_Position = vec4(pos, 0, 1);
	
	fs_in_tc.x = (gl_VertexID / 2) * 2.0;
	fs_in_tc.y = 1.0 - (gl_VertexID % 2) * 2.0;
}