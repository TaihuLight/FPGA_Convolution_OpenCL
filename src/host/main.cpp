// main.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "CL/opencl.h"
#include "../../inc/AOCLUtils/aocl_utils.h"
// #define DEBUG
using namespace aocl_utils;

#define BLOCK_SIZE 64
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements

scoped_array<cl_mem> input_a_buf; // num_devices elements
scoped_array<cl_mem> input_b_buf; // num_devices elements
scoped_array<cl_mem> output_buf; // num_devices elements

scoped_array<scoped_aligned_ptr<float> > input_a; // num_devices elements
scoped_aligned_ptr<float> input_b;
scoped_array<scoped_aligned_ptr<float> > output; // num_devices elements

scoped_array<float> ref_output;
scoped_array<float> arch1_output;
scoped_array<unsigned> rows_per_device; // num_devices elements

// Problem data.
unsigned A_height = 32 * BLOCK_SIZE;
unsigned A_width  = 16 * BLOCK_SIZE;
const unsigned &B_height = A_width;
unsigned B_width  = 16 * BLOCK_SIZE;
const unsigned &C_height = A_height;
const unsigned &C_width  = B_width;

// Conv parameters
unsigned conv_engine = 1; // 0 - direct(default), 1 - matmult, 2 - winograd 
unsigned img_w, img_h, pad_w, pad_h, stride_w, stride_h, output_h, output_w, kernel_size;
const unsigned kernel_n = 2, kernel_h = 3, kernel_w = 3;
unsigned kernels[kernel_n][kernel_h][kernel_w], *krnls_row_buf, *img;
float *out_buf, *col_buf; 
bool is_gemm_conv = false;
#ifdef TILED
char *kernel_path = "/home/xmeng/code/fpga_conv/build/matrix_mult_tiled.aocx";
#else
char *kernel_path = "/home/xmeng/code/fpga_conv/build/matrix_mult.aocx";
#endif
float rand_float();
bool init_opencl(char *binary_name);
bool init_matMult();
void run_matMult();
void cleanup();
// void run();
void compute_reference();
void verify();
void cleanup();
void init_conv();
void im2col();
void kernel2row();
void print();
int main(int argc, char **argv)
{
  Options options(argc, argv);
  if(options.has("conv"))
  {
    conv_engine = options.get<unsigned>("conv");
    if(options.has("img_w"))
    {
      img_w = options.get<unsigned>("img_w");
    }
    if(options.has("img_h"))
    {
      img_h = options.get<unsigned>("img_h");
    }
    if(options.has("pad_w"))
    {
      pad_w = options.get<unsigned>("pad_w");
    }
    if(options.has("pad_h"))
    {
      pad_h = options.get<unsigned>("pad_h");
    }
    if(options.has("stride_w"))
    {
      stride_w = options.get<unsigned>("stride_w");
    }
    if(options.has("stride_h"))
    {
      stride_h = options.get<unsigned>("stride_h");
    }
    printf("GEMM convolution process:\nImage size: %d * %d, Kernel size: %d * %d * %d\n", img_h, img_w, kernel_n, kernel_h, kernel_w);
    // Preparation of kernels
    kernels[0][0][0] = 0;
    kernels[0][0][1] = 1;
    kernels[0][0][2] = 0;
    kernels[0][1][0] = 1;
    kernels[0][1][1] = 0;
    kernels[0][1][2] = 1;
    kernels[0][2][0] = 0;
    kernels[0][2][1] = 1;
    kernels[0][2][2] = 0;
    
    kernels[1][0][0] = 1;
    kernels[1][0][1] = 1;
    kernels[1][0][2] = 1;
    kernels[1][1][0] = 1;
    kernels[1][1][1] = 1;
    kernels[1][1][2] = 1;
    kernels[1][2][0] = 1;
    kernels[1][2][1] = 1;
    kernels[1][2][2] = 1;

    output_h = (img_h - kernel_h) / stride_h + 1;
    output_w = (img_w - kernel_w) / stride_w + 1;
    kernel_size = kernel_h * kernel_w; 

    init_conv();
    A_height = kernel_n;
    A_width = kernel_size;
    B_width = output_h * output_w;
    is_gemm_conv = true;
  }
  else 
  {
    if(options.has("ah")) {
      A_height = options.get<unsigned>("ah");
    }
    if(options.has("aw")) {
      A_width = options.get<unsigned>("aw");
    }
    if(options.has("bw")) {
      B_width = options.get<unsigned>("bw");
    }
  }
  printf("Matrix sizes:\n  A: %d x %d\n  B: %d x %d\n  C: %d x %d\n",
      A_height, A_width, B_height, B_width, C_height, C_width);
  // print();
  // Spot check matrix sizes. They all must be a multiple of BLOCK_SIZE,
  // although it is relatively straightforward to handle non-multiples
  // by adding padding. For simplicity, this example does not pad.
  if((A_height % BLOCK_SIZE) != 0 || (A_width % BLOCK_SIZE) != 0 ||
     (B_height % BLOCK_SIZE) != 0 || (B_width % BLOCK_SIZE) != 0 ||
     (C_height % BLOCK_SIZE) != 0 || (C_width % BLOCK_SIZE) != 0) {
    printf("Matrix sizes must be a multiple of %d. Perform padding if I am doing convolution: %s\n", BLOCK_SIZE, is_gemm_conv ? "YES" : "NO");
    if(!is_gemm_conv) return -1;
    // Perform padding for convoultion
    int new_ah = A_height, new_aw = A_width, new_bh = B_height, new_bw = B_width, new_ch = C_height, new_cw = C_width;
    int q;
    if((A_height % BLOCK_SIZE) != 0)
    {
      q = A_height / BLOCK_SIZE;
      new_ah = (q + 1) * BLOCK_SIZE;
      new_ch = new_ah;
      // printf("New A_height:%d, A_width:%d\n", A_height, A_width);
    }
    if((A_width % BLOCK_SIZE) != 0)
    {
      q = A_width / BLOCK_SIZE;
      new_aw = (q + 1) * BLOCK_SIZE;
      new_bh = (q + 1) * BLOCK_SIZE;
      // printf("New B_height:%d, B_width:%d\n", B_height, B_width);
    }
    if((B_width % BLOCK_SIZE) != 0)
    {
      q = B_width / BLOCK_SIZE;
      new_bw = (q + 1) * BLOCK_SIZE;
      new_cw = new_bw;
      // printf("New C_height:%d, C_width:%d\n", C_height, C_width);
    }

    // krnls_row_buf = (unsigned*)realloc(krnls_row_buf, new_ah * new_bh * sizeof(unsigned));
    // memset(krnls_row_buf + A_height * A_width, 0, (new_ah * new_aw - A_height * A_width) * sizeof(float));
    // A_height = new_ah; A_width = new_aw;
    // col_buf = (float*)realloc(col_buf, new_bh * new_bw * sizeof(float));
    // memset(col_buf + B_height *B_width, 0, (new_bh * new_bw - B_height * B_width) * sizeof(float));
    // B_width = new_bw;
    // out_buf = (float*)realloc(out_buf, new_ch * new_cw * sizeof(float));
    // memset(out_buf, 0, new_ch * new_cw * sizeof(float));

    unsigned *tmp_k = krnls_row_buf;
    float *tmp_c = col_buf, *tmp_o = out_buf;
    krnls_row_buf = (unsigned *)malloc(new_ah * new_aw * sizeof(unsigned));
    memset(krnls_row_buf, 0, new_ah * new_aw * sizeof(unsigned));
    
    for(int i = 0; i < A_height; ++i)
    {
      for(int j = 0; j < A_width; ++j)
      {
        krnls_row_buf[i * new_aw + j] = tmp_k[i * A_width + j];
      }
    }
    
  #ifdef DEBUG
  printf("Matrix A:\n");
  for(int i = 0; i < A_height; ++i)
  {
    for(int j = 0; j < A_width; ++j)
    {
      printf("%d ", krnls_row_buf[i * A_width + j]);
    }
    printf("\n");
  }
  #endif
    col_buf = (float *)malloc(new_bh * new_bw * sizeof(float));
    memset(col_buf, 0, new_bh * new_bw * sizeof(float));
    
    for(int i = 0; i < B_height; ++i)
    {
      for(int j = 0; j < B_width; ++j)
      {
        col_buf[i * new_bw + j] = tmp_c[i * B_width + j];
      }
    } 
    A_height = new_ah; A_width = new_aw;
B_width = new_bw;
    out_buf = (float *)malloc(new_ch * new_cw * sizeof(float));
    memset(out_buf, 0, new_ch * new_cw * sizeof(float));
    delete[] tmp_k, tmp_c, tmp_o;
    printf("New Matrix sizes:\n  A: %d x %d\n  B: %d x %d\n  C: %d x %d\n",
      A_height, A_width, B_height, B_width, C_height, C_width);
  }

  if (!init_opencl(kernel_path))
    return 0;

  run_matMult();
  compute_reference();
  verify();
  // print();
  cleanup();
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

bool init_opencl(char *binary_file)
{
	cl_int status;

	printf("Initializing OpenCL...\n");

	if (!setCwdToExeDir()) // Sets the current working directory to be the same as the directory
	// containing the running executable.
	{
		return false;
	}
    
    // Get the OpenCL platform.
    platform = findPlatform("Altera SDK for OpenCL");
    if(platform == NULL) 
    {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device.
    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);
    for(unsigned i = 0; i < num_devices; ++i) {
        printf("  %s\n", getDeviceName(device[i]).c_str());
    }

    // Create the context.
    context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the program for all device. Use the first device as the
    // representative device (assuming all device are of the same type).
    printf("Using AOCX: %s\n", binary_file);
    program = createProgramFromBinary(context, binary_file, device, num_devices);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create per-device objects.
    queue.reset(num_devices);
    kernel.reset(num_devices);
    rows_per_device.reset(num_devices); 

    input_a_buf.reset(num_devices);
    input_b_buf.reset(num_devices);
    output_buf.reset(num_devices);

  // for(unsigned i = 0; i < num_devices; ++i) {
  //   // Command queue.
  //   queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
  //   checkError(status, "Failed to create command queue");

  //   // Kernel.
  //   kernel[i] = clCreateKernel(program, kernel_name, &status);
  //   checkError(status, "Failed to create kernel");

  //   // Determine the number of rows processed by this device.
  //   // First do this computation in block-rows.
  //   rows_per_device[i] = num_block_rows / num_devices; // this is the number of block-rows

  //   // Spread out the remainder of the block-rows over the first
  //   // N % num_devices.
  //   if(i < (num_block_rows % num_devices)) {
  //     rows_per_device[i]++;
  //   }

  //   // Multiply by BLOCK_SIZE to get the actual number of rows.
  //   rows_per_device[i] *= BLOCK_SIZE;

  //   // Input buffers.
  //   // For matrix A, each device only needs the rows corresponding
  //   // to the rows of the output matrix. We specifically
  //   // assign this buffer to the first bank of global memory.
  //   input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA,
  //       rows_per_device[i] * A_width * sizeof(float), NULL, &status);
  //   checkError(status, "Failed to create buffer for input A");

  //   // For matrix B, each device needs the whole matrix. We specifically
  //   // assign this buffer to the second bank of global memory.
  //   input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA,
  //       B_height * B_width * sizeof(float), NULL, &status);
  //   checkError(status, "Failed to create buffer for input B");

  //   // Output buffer. This is matrix C, for the rows that are computed by this
  //   // device. We assign this buffer to the first bank of global memory,
  //   // although it is not material to performance to do so because
  //   // the reads from the input matrices are far more frequent than the
  //   // write to the output matrix.
  //   output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA,
  //       rows_per_device[i] * C_width * sizeof(float), NULL, &status);
  //   checkError(status, "Failed to create buffer for output");

  //   status = clGetDeviceInfo(
  //     device[i],
  //     CL_DEVICE_SVM_CAPABILITIES,
  //     sizeof(cl_device_svm_capabilities),
  //     &caps,
  //     0
  //   );
  //   checkError(status, "Failed to get device info");

  //   if (!(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
  //     printf("The host was compiled with USE_SVM_API, however the device currently being targeted does not support SVM.\n");
  //     // Free the resources allocated
  //     cleanup();
  //     return false;
  //   }

  //   return true;
  // }
    return init_matMult();
} 

bool init_matMult()
{
  cl_int status;
  const unsigned num_block_rows = C_height / BLOCK_SIZE;
  printf("Num of devices:%d\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    kernel[i] = clCreateKernel(program, "matrixMult", &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of rows processed by this device.
    // First do this computation in block-rows.
    rows_per_device[i] = num_block_rows / num_devices; // this is the number of block-rows

    // Spread out the remainder of the block-rows over the first
    // N % num_devices.
    if(i < (num_block_rows % num_devices)) {
      rows_per_device[i]++;
    }

    // Multiply by BLOCK_SIZE to get the actual number of rows.
    rows_per_device[i] *= BLOCK_SIZE;

    // Input buffers.
    // For matrix A, each device only needs the rows corresponding
    // to the rows of the output matrix. We specifically
    // assign this buffer to the first bank of global memory.
    input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA,
        rows_per_device[i] * A_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    // For matrix B, each device needs the whole matrix. We specifically
    // assign this buffer to the second bank of global memory.
    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA,
        B_height * B_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer. This is matrix C, for the rows that are computed by this
    // device. We assign this buffer to the first bank of global memory,
    // although it is not material to performance to do so because
    // the reads from the input matrices are far more frequent than the
    // write to the output matrix.
    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA,
        rows_per_device[i] * C_width * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");



  }
  // Data init
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  // Generate input matrices A and B. For matrix A, we divide up the host
  // buffers so that the buffers are aligned for each device. The whole of
  // matrix B is used by each device, so it does not need to be divided.
  printf("Generating input matrices\n");
  input_a.reset(num_devices);
  output.reset(num_devices);

  if(!is_gemm_conv)
  {
    for(unsigned i = 0; i < num_devices; ++i) {
      input_a[i].reset(rows_per_device[i] * A_width);
      output[i].reset(rows_per_device[i] * C_width);

      for(unsigned j = 0; j < rows_per_device[i] * A_width; ++j) {
        input_a[i][j] = rand_float();
      }
    }

    input_b.reset(B_height * B_width);
    for(unsigned i = 0; i < B_height * B_width; ++i) {
      input_b[i] = rand_float();
    }
  }
  else
  {
    int count = 0;
    for(unsigned i = 0; i < num_devices; ++i)
    {
      input_a[i].reset(rows_per_device[i] * A_width);
      output[i].reset(rows_per_device[i] * C_width);

      for(unsigned j = 0; j < rows_per_device[i] * A_width; ++j)
      {
        input_a[i][j] = krnls_row_buf[count++];
      }
    }

    input_b.reset(B_height * B_width);
    for(unsigned i = 0; i < B_height * B_width; ++i)
    {
      input_b[i] = col_buf[i];
    }
  }
  // for(unsigned i = 0; i < 16; ++i) {
  //   printf("input_b[%d]=%f\n", i, input_b[i]);
  // }
  return true;
}

void init_conv()
{
  // TODO: initialize *img, *out_buf, *col_buf, *krnls_row_buf
  img = new unsigned[img_w * img_h];
  // out_buf = new float[output_h * output_w * kernel_n];
  // col_buf = new float[kernel_size * output_h * output_w];
  // krnls_row_buf = new unsigned[kernel_n * kernel_size];

  out_buf = (float*)malloc(output_h * output_w * kernel_n * sizeof(float));
  col_buf = (float*)malloc(kernel_size * output_h * output_w * sizeof(float));
  krnls_row_buf = (unsigned*)malloc(kernel_n * kernel_size * sizeof(unsigned));
  srand(time(NULL));
  printf("Randomly initializing IMAGE...\n");
  for (int h = 0; h < img_h; ++h)
  {
    for (int w = 0; w < img_w; ++w)
    {
      img[h * img_w + w] = (unsigned)rand() % 256;
      #ifdef DEBUG
      printf("%d ", img[h * img_w + w]);
      #endif
    }
    #ifdef DEBUG
    printf("\n");
    #endif
  }

  memset(out_buf, 0, output_h * output_w * kernel_n * sizeof(float));
  memset(col_buf, 0, kernel_size * output_h * output_w * sizeof(float));
  memset(krnls_row_buf, 0, kernel_n * kernel_size * sizeof(unsigned));

  im2col();
  kernel2row();
  // Finally two matrices are needed to be multiplied: krnls_row_buf and col_buf
} 

void im2col()
{
  int count = 0;
  for (int h = 0; h < output_h; h += stride_h)
  {
      for (int w = 0; w < output_w; w += stride_w)
      {
          
          for (int krnl_h = 0; krnl_h < kernel_h; ++krnl_h)
          {
              for (int krnl_w = 0; krnl_w < kernel_w; ++krnl_w)
              {

                  *(col_buf + (krnl_h * kernel_w + krnl_w) * output_h * output_w + count) 
                      = img[(h + krnl_h) * img_w + w + krnl_w];
                      // = data_buf[img_h + krnl_h][img_w + krnl_w];
              }
          }
          ++count;
      }
  }
#ifdef DEBUG
  printf("After im2col operation:\n");
  for(int i = 0; i < kernel_size; ++i)
  {
    for(int j = 0; j < output_w * output_h; ++j)
    {
      printf("%3.0f ", col_buf[i * output_w * output_h + j]);
    }
    printf("\n");
  }
  #endif
}

void kernel2row()
{
  for (int krnl_n = 0; krnl_n < kernel_n; ++krnl_n)
  {
      memcpy(krnls_row_buf + krnl_n * kernel_size, kernels[krnl_n], sizeof(unsigned) * kernel_size);
  }
#ifdef DEBUG
  printf("Matrix A:\n");
  for(int i = 0; i < kernel_n; ++i)
  {
    for(int j = 0; j < kernel_size; ++j)
    {
      printf("%d ", krnls_row_buf[i * kernel_size + j]);
    }
    printf("\n");
  }
  #endif
}

void run_matMult()
{
  cl_int status;
  // Transfer inputs to each device. Each of the host buffers supplied to
  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  for(unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, rows_per_device[i] * A_width * sizeof(float), input_a[i], 0, NULL, NULL);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, B_width * B_height * sizeof(float), input_b, 0, NULL, NULL);
    checkError(status, "Failed to transfer input B");
  }

  // Wait for all queues to finish.
  for(unsigned i = 0; i < num_devices; ++i) {
    clFinish(queue[i]);
  }

  // Launch kernels.
  // This is the portion of time that we'll be measuring for throughput
  // benchmarking.
  scoped_array<cl_event> kernel_event(num_devices);

  const double start_time = getCurrentTimestamp();
  for(unsigned i = 0; i < num_devices; ++i) 
  {
    // Set kernel arguments.
    unsigned argi = 0;
    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);
#ifdef TILED
    status = clSetKernelArg(kernel[i], argi++, sizeof(A_height), &A_height);
    checkError(status, "Failed to set argument %d", argi - 1);
#endif
    status = clSetKernelArg(kernel[i], argi++, sizeof(A_width), &A_width);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(B_width), &B_width);
    checkError(status, "Failed to set argument %d", argi - 1);


    // Enqueue kernel.
    // Use a global work size corresponding to the size of the output matrix.
    // Each work-item computes the result for one value of the output matrix,
    // so the global work size has the same dimensions as the output matrix.
    //
    // The local work size is one block, so BLOCK_SIZE x BLOCK_SIZE.
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size[2] = {C_width, rows_per_device[i]};
    const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};
    printf("Launching for device %d (global size: %zd, %zd)\n", i, global_work_size[0], global_work_size[1]);

    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 2, NULL,
        global_work_size, local_work_size, 0, NULL, &kernel_event[i]);
    checkError(status, "Failed to launch kernel");


  // Wait for all kernels to finish.
  clWaitForEvents(num_devices, kernel_event);

  // Release kernel events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
  }

  // Read the result.
  for(unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_TRUE,
        0, rows_per_device[i] * C_width * sizeof(float), output[i], 0, NULL, NULL);
    checkError(status, "Failed to read output matrix");
  }
  printf("Finish the matMult function.\n");
}
}

void compute_reference() {
  // Compute the reference output.
  printf("Computing reference output\n");
  ref_output.reset(C_height * C_width);
    if(!is_gemm_conv) {
    for(unsigned y = 0, dev_index = 0; y < C_height; ++dev_index) {
      for(unsigned yy = 0; yy < rows_per_device[dev_index]; ++yy, ++y) {
        for(unsigned x = 0; x < C_width; ++x) {
          // Compute result for C(y, x)
          float sum = 0.0f;
          for(unsigned k = 0; k < A_width; ++k) {
            sum += input_a[dev_index][yy * A_width + k] * input_b[k * B_width + x];
          }
          ref_output[y * C_width + x] = sum;
        }
      }
    }
  }
  else {
     for (int i = 0; i < C_width * C_height; ++i)
  {
    ref_output[i] = 0;
  }
  float *temp = new float[kernel_n];
  int count = 0;
  for (int h = 0; h < output_h; h += stride_h)
  {
      for (int w = 0; w < output_w; w += stride_w)
      {
          memset(temp, (float)0, sizeof(float) * kernel_n);
          for (int krnl_h = 0; krnl_h < kernel_h; ++krnl_h)
          {
              for (int krnl_w = 0; krnl_w < kernel_w; ++krnl_w)
              {
                  for (int ch = 0; ch < kernel_n; ++ch)
                  {
                      temp[ch] += kernels[ch][krnl_h][krnl_w] * img[(h + krnl_h) * img_w + w + krnl_w];
                  }
              }
          }
          for (int ch = 0; ch < kernel_n; ++ch)
          {
              ref_output[ch * C_width + output_w * h + w/*count++*/] = temp[ch];
          }
      }
  }
 
}
}

void verify() {
  printf("Verifying\n");

  // Compute the L^2-Norm of the difference between the output and reference
  // output matrices and compare it against the L^2-Norm of the reference.
  float diff = 0.0f;
  float ref = 0.0f;
  for(unsigned y = 0, dev_index = 0; y < C_height; ++dev_index) {
    for(unsigned yy = 0; yy < rows_per_device[dev_index]; ++yy, ++y) {
      for(unsigned x = 0; x < C_width; ++x) {
        const float o = output[dev_index][yy * C_width + x];
        const float r = ref_output[y * C_width + x];
        const float d = o - r;
        diff += d * d;
        ref += r * r;
      }
    }
  }

  const float diff_l2norm = sqrtf(diff);
  const float ref_l2norm = sqrtf(ref);
  const float error = diff_l2norm / ref_l2norm;
  const bool pass = error < 1e-6;
  printf("Verification: %s\n", pass ? "PASS" : "FAIL");
  if(!pass) {
    printf("Error (L^2-Norm): %0.3g\n", error);
  }
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
#if USE_SVM_API == 0
    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }
#else
    if(input_a[i].get())
      input_a[i].reset();
    if(output[i].get())
      output[i].reset();
#endif /* USE_SVM_API == 0 */
  }
#if USE_SVM_API == 1
  if(input_b.get())
    input_b.reset();
#endif /* USE_SVM_API == 1 */

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

void print()
{
  printf("Matrix A:\n");
  for(int i = 0; i < A_height; ++i)
  {
    for(int j = 0; j < A_width; ++j)
    {
      printf("%d ", krnls_row_buf[i * A_width + j]);
    }
    printf("\n");
  }

  printf("Matrix B:\n");
  for(int i = 0; i < B_height; ++i)
  {
    for(int j = 0; j < B_width; ++j)
    {
      printf("%3.0f ", col_buf[i * B_width + j]);
    }
    printf("\n");
  }

  printf("Matrix C:\n");
  for(int i = 0; i < C_height; ++i)
  {
    for(int j = 0; j < C_width; ++j)
    {
      printf("%3.0f ", output[0][i * C_width + j]);
    }
    printf("\n");
  }

  printf("Matrix Ref:\n");
  for(int i = 0; i < C_height; ++i)
  {
    for(int j = 0; j < C_width; ++j)
    {
      printf("%3.0f ", ref_output[i * C_width + j]);
    }
    printf("\n");
  }


}