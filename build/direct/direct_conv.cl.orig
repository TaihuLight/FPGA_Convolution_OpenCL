__kernel void conv16*16(__global const float4 *in, __constant float *krnl, __gloabl float4 *out, int out_w)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	// Kernel size is 3*3
	float4 sum = 0.0f;
	// Check if out-of-bound
	if(x < 0 || x > 13 || y < 0 || y > 13)
	{
		return;
	}
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			sum += in[(x + i) * 16 + y + j] * krnl[i * 3 + j];
		}
	}

	out[x * out_W + j] = sum;
}