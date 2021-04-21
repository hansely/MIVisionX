#include <kernels.h>

struct LocalData {
    NeuralNetworkCommonHandle * handle;
    int in_offset = 0;
    int out_offset = 0;
    int starts_offset = 0;
    int ends_offset = 0;
    int axes_offset = 0;
    int steps_offset = 0;
    cl_kernel slice_kernel;
    size_t slice_global[3];
};

static vx_status VX_CALLBACK validateSliceLayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) {
    //get input params
    vx_enum type, starts_type, ends_type, axes_type, steps_type, out_type;
    vx_size num_dims, starts_dims, ends_dims, axes_dims, steps_dims, out_num_dims;
    vx_size input_dims[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &starts_dims, sizeof(starts_dims)));   
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &starts_type, sizeof(starts_type)));
    if ((starts_type != VX_TYPE_INT32) && (starts_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, &ends_dims, sizeof(ends_dims)));   
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &ends_type, sizeof(ends_type)));
    if ((ends_type != VX_TYPE_INT32) && (ends_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    
    if (parameters[4]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, &axes_dims, sizeof(axes_dims)));   
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &axes_type, sizeof(axes_type)));
        if ((axes_type != VX_TYPE_INT32) && (axes_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    }
    
    if (parameters[5]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, &steps_dims, sizeof(steps_dims)));   
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &steps_type, sizeof(steps_type)));
        if ((steps_type != VX_TYPE_INT32) && (steps_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    }

    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeSliceLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    //get input params
    vx_size num_of_dims, params_dim;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    vx_size input_dims[num_of_dims];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));    
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &params_dim, sizeof(params_dim)));
    vx_size params_dims[params_dim];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, params_dims, sizeof(params_dims)));


    int element_count = params_dims[0];    
    int max_dim[3] = {0,0,0};

    for (int i=0; i<num_of_dims; i++) {
        max_dim[i] = input_dims[i];
    }

    LocalData * data = new LocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    // get cl buffer, stride, and offsets
    cl_mem input_mem = nullptr, output_mem = nullptr, starts_mem = nullptr, ends_mem = nullptr, axes_mem = nullptr, steps_mem = nullptr;
    vx_size in_stride[4] = {1,1,1,1}, out_stride[4] = {1,1,1,1}, starts_stride[4] = {1,1,1,1}, ends_stride[4] = {1,1,1,1}, axes_stride[4] = {1,1,1,1}, steps_stride[4] = {1,1,1,1};

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &input_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_STRIDE_OPENCL, in_stride, sizeof(in_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_OFFSET_OPENCL, &data->in_offset, sizeof(vx_size)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &output_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_STRIDE_OPENCL, out_stride, sizeof(out_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_OFFSET_OPENCL, &data->out_offset, sizeof(vx_size)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &starts_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_STRIDE_OPENCL, starts_stride, sizeof(starts_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_OFFSET_OPENCL, &data->starts_offset, sizeof(vx_size)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_OPENCL, &ends_mem, sizeof(cl_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_STRIDE_OPENCL, ends_stride, sizeof(ends_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_OFFSET_OPENCL, &data->ends_offset, sizeof(vx_size)));

    if(parameters[4]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &axes_mem, sizeof(cl_mem)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_STRIDE_OPENCL, axes_stride, sizeof(axes_stride)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_OFFSET_OPENCL, &data->axes_offset, sizeof(vx_size)));
    }
    if(parameters[5]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_OPENCL, &steps_mem, sizeof(cl_mem)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_STRIDE_OPENCL, steps_stride, sizeof(steps_stride)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_OFFSET_OPENCL, &data->steps_offset, sizeof(vx_size)));
    }
    
    data->in_offset >>= 2;
    data->out_offset >>= 2;
    data->starts_offset >>= 2;
    data->steps_offset >>= 2;

    for (int i=0; i<4; i++) {
        in_stride[i] >>= 2;
        out_stride[i] >>= 2;
        starts_stride[i] >>= 2;
        ends_stride[i] >>= 2;
        axes_stride[i] >>= 2;
        steps_stride[i] >>= 2;
    }

    char item[8192];
    if (num_of_dims) {
        if (type == VX_TYPE_FLOAT32) {
            if(parameters[5]) {
                sprintf(item,
                    "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                    "__kernel void slice(__global float * in, __global float * out, __global int * starts, __global int * ends, __global int * axes, __global int * steps)\n"
                    "{ \n"
                    "   uint x = get_global_id(0);\n"
                    "   uint y = get_global_id(1);\n"
                    "   uint c = get_global_id(2);\n"
                    "   int in_stride[4] = {%d, %d, %d, %d};\n"
                    "   int out_stride[4] = {%d, %d, %d, %d};\n"
                    "   int starts_stride = %d;\n"
                    "   int steps_stride = %d;\n"
                    "   int in_offset = %d;\n"
                    "   int out_offset = %d;\n"
                    "   int starts_offset = %d;\n"
                    "   int steps_offset = %d;\n"
                    "   int num_dim = %d;\n"
                    "   int dim1 = num_dim - 1;\n"
                    "   int dim2 = num_dim - 2;\n"
                    "   int dim3 = num_dim - 3;\n"
                    "   int max_dim1 = %d;\n"
                    "   int max_dim2 = %d;\n"
                    "   int max_dim3 = %d;\n"
                    "   int start1 = *(__global int*)&starts[starts_offset + dim1*starts_stride];\n"
                    "   int start2 = *(__global int*)&starts[starts_offset + dim2*starts_stride];\n"
                    "   int start3 = *(__global int*)&starts[starts_offset + dim3*starts_stride];\n"
                    "   if (start1 > (max_dim1-1))\n"
                    "       start1 = max_dim1-1;\n"
                    "   else if (start1 < 0)\n"
                    "       start1 = max_dim1 + start1;\n"
                    "   if (start2 > (max_dim2-1))\n"
                    "       start2 = max_dim2-1;\n"
                    "   else if (start2 < 0)\n"
                    "       start2 = max_dim2 + start2;\n"
                    "   if (start3 > (max_dim3-1))\n"
                    "       start3 = max_dim3-1;\n"
                    "   else if (start3 < 0)\n"
                    "       start3 = max_dim3 + start3;\n"
                    
                    "   int index1 = start1 + *(__global int*)&steps[steps_offset + dim1*steps_stride] * x;\n"
                    "   int index2 = start2 + *(__global int*)&steps[steps_offset + dim2*steps_stride] * y;\n"
                    "   int index3 = start3 + *(__global int*)&steps[steps_offset + dim3*steps_stride] * c;\n"
                    "   index1 = (dim1 >= 0) ? index1 : 0;\n"
                    "   index2 = (dim2 >= 0) ? index2 : 0;\n"
                    "   index3 = (dim3 >= 0) ? index3 : 0;\n"
                    "   float value;\n"
                    "   uint offset;\n"
                    "   value = *(__global float*)&in[in_offset + index1*in_stride[0] + index2*in_stride[1] + index3*in_stride[2]];\n"
                    "   offset = out_offset + x*out_stride[0] + y*out_stride[1] + c*out_stride[2];\n"
                    "   out += offset;\n"
                    "   *(__global float *)&out[0] = value;\n"
                    "}\n", (int)in_stride[0], (int)in_stride[1], (int)in_stride[2], (int)in_stride[3], (int)out_stride[0], (int)out_stride[1], (int)out_stride[2], (int)out_stride[3], (int)starts_stride[0], (int)steps_stride[0], data->in_offset, data->out_offset, data->starts_offset, data->steps_offset, (int)num_of_dims, max_dim[0], max_dim[1], max_dim[2]);
            }
            else {
                sprintf(item,
                    "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                    "__kernel void slice(__global float * in, __global float * out, __global int * starts, __global int * ends)\n"
                    "{ \n"
                    "   uint x = get_global_id(0);\n"
                    "   uint y = get_global_id(1);\n"
                    "   uint c = get_global_id(2);\n"
                    "   int in_stride[4] = {%d, %d, %d, %d};\n"
                    "   int out_stride[4] = {%d, %d, %d, %d};\n"
                    "   int starts_stride = %d;\n"
                    "   int in_offset = %d;\n"
                    "   int out_offset = %d;\n"
                    "   int starts_offset = %d;\n"
                    "   int num_dim = %d;\n"
                    "   int dim1 = num_dim - 1;\n"
                    "   int dim2 = num_dim - 2;\n"
                    "   int dim3 = num_dim - 3;\n"
                    "   int max_dim1 = %d;\n"
                    "   int max_dim2 = %d;\n"
                    "   int max_dim3 = %d;\n"
                    "   int start1 = *(__global int*)&starts[starts_offset + dim1*starts_stride];\n"
                    "   int start2 = *(__global int*)&starts[starts_offset + dim2*starts_stride];\n"
                    "   int start3 = *(__global int*)&starts[starts_offset + dim3*starts_stride];\n"
                    "   if (start1 > (max_dim1-1))\n"
                    "       start1 = max_dim1-1;\n"
                    "   else if (start1 < 0)\n"
                    "       start1 = max_dim1 + start1;\n"
                    "   if (start2 > (max_dim2-1))\n"
                    "       start2 = max_dim2-1;\n"
                    "   else if (start2 < 0)\n"
                    "       start2 = max_dim2 + start2;\n"
                    "   if (start3 > (max_dim3-1))\n"
                    "       start3 = max_dim3-1;\n"
                    "   else if (start3 < 0)\n"
                    "       start3 = max_dim3 + start3;\n"
                    
                    "   int index1 = start1 + x;\n"
                    "   int index2 = start2 + y;\n"
                    "   int index3 = start3 + c;\n"
                    "   index1 = (dim1 >= 0) ? index1 : 0;\n"
                    "   index2 = (dim2 >= 0) ? index2 : 0;\n"
                    "   index3 = (dim3 >= 0) ? index3 : 0;\n"
                    "   float value;\n"
                    "   uint offset;\n"
                    "   value = *(__global float*)&in[in_offset + index1*in_stride[0] + index2*in_stride[1] + index3*in_stride[2]];\n"
                    "   offset = out_offset + x*out_stride[0] + y*out_stride[1] + c*out_stride[2];\n"
                    "   out += offset;\n"
                    "   *(__global float *)&out[0] = value;\n"
                    "}\n", (int)in_stride[0], (int)in_stride[1], (int)in_stride[2], (int)in_stride[3], (int)out_stride[0], (int)out_stride[1], (int)out_stride[2], (int)out_stride[3], (int)starts_stride[0], data->in_offset, data->out_offset, data->starts_offset, (int)num_of_dims, max_dim[0], max_dim[1], max_dim[2]);
            }
        }
        else if (type == VX_TYPE_FLOAT16) {
            if(parameters[5]) {
                sprintf(item,
                    "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                    "__kernel void slice(__global half * in, __global half * out, __global int * starts, __global int * ends, __global int * axes, __global int * steps)\n"
                    "{ \n"
                    "   uint x = get_global_id(0);\n"
                    "   uint y = get_global_id(1);\n"
                    "   uint c = get_global_id(2);\n"
                    "   int in_stride[4] = {%d, %d, %d, %d};\n"
                    "   int out_stride[4] = {%d, %d, %d, %d};\n"
                    "   int starts_stride = %d;\n"
                    "   int steps_stride = %d;\n"
                    "   int in_offset = %d;\n"
                    "   int out_offset = %d;\n"
                    "   int starts_offset = %d;\n"
                    "   int steps_offset = %d;\n"
                    "   int num_dim = %d;\n"
                    "   int dim1 = num_dim - 1;\n"
                    "   int dim2 = num_dim - 2;\n"
                    "   int dim3 = num_dim - 3;\n"
                    "   int max_dim1 = %d;\n"
                    "   int max_dim2 = %d;\n"
                    "   int max_dim3 = %d;\n"
                    "   int start1 = *(__global int*)&starts[starts_offset + dim1*starts_stride];\n"
                    "   int start2 = *(__global int*)&starts[starts_offset + dim2*starts_stride];\n"
                    "   int start3 = *(__global int*)&starts[starts_offset + dim3*starts_stride];\n"
                    "   if (start1 > (max_dim1-1))\n"
                    "       start1 = max_dim1-1;\n"
                    "   else if (start1 < 0)\n"
                    "       start1 = max_dim1 + start1;\n"
                    "   if (start2 > (max_dim2-1))\n"
                    "       start2 = max_dim2-1;\n"
                    "   else if (start2 < 0)\n"
                    "       start2 = max_dim2 + start2;\n"
                    "   if (start3 > (max_dim3-1))\n"
                    "       start3 = max_dim3-1;\n"
                    "   else if (start3 < 0)\n"
                    "       start3 = max_dim3 + start3;\n"
                    
                    "   int index1 = start1 + *(__global int*)&steps[steps_offset + dim1*steps_stride] * x;\n"
                    "   int index2 = start2 + *(__global int*)&steps[steps_offset + dim2*steps_stride] * y;\n"
                    "   int index3 = start3 + *(__global int*)&steps[steps_offset + dim3*steps_stride] * c;\n"
                    "   index1 = (dim1 >= 0) ? index1 : 0;\n"
                    "   index2 = (dim2 >= 0) ? index2 : 0;\n"
                    "   index3 = (dim3 >= 0) ? index3 : 0;\n"
                    "   half value;\n"
                    "   uint offset;\n"
                    "   value = *(__global half*)&in[in_offset + index1*in_stride[0] + index2*in_stride[1] + index3*in_stride[2]];\n"
                    "   offset = out_offset + x*out_stride[0] + y*out_stride[1] + c*out_stride[2];\n"
                    "   out += offset;\n"
                    "   *(__global half *)&out[0] = value;\n"
                    "}\n", (int)in_stride[0], (int)in_stride[1], (int)in_stride[2], (int)in_stride[3], (int)out_stride[0], (int)out_stride[1], (int)out_stride[2], (int)out_stride[3], (int)starts_stride[0], (int)steps_stride[0], data->in_offset, data->out_offset, data->starts_offset, data->steps_offset, (int)num_of_dims, max_dim[0], max_dim[1], max_dim[2]);
            }
            else {
                sprintf(item,
                    "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                    "__kernel void slice(__global half * in, __global half * out, __global int * starts, __global int * ends)\n"
                    "{ \n"
                    "   uint x = get_global_id(0);\n"
                    "   uint y = get_global_id(1);\n"
                    "   uint c = get_global_id(2);\n"
                    "   int in_stride[4] = {%d, %d, %d, %d};\n"
                    "   int out_stride[4] = {%d, %d, %d, %d};\n"
                    "   int starts_stride = %d;\n"
                    "   int in_offset = %d;\n"
                    "   int out_offset = %d;\n"
                    "   int starts_offset = %d;\n"
                    "   int num_dim = %d;\n"
                    "   int dim1 = num_dim - 1;\n"
                    "   int dim2 = num_dim - 2;\n"
                    "   int dim3 = num_dim - 3;\n"
                    "   int max_dim1 = %d;\n"
                    "   int max_dim2 = %d;\n"
                    "   int max_dim3 = %d;\n"
                    "   int start1 = *(__global int*)&starts[starts_offset + dim1*starts_stride];\n"
                    "   int start2 = *(__global int*)&starts[starts_offset + dim2*starts_stride];\n"
                    "   int start3 = *(__global int*)&starts[starts_offset + dim3*starts_stride];\n"
                    "   if (start1 > (max_dim1-1))\n"
                    "       start1 = max_dim1-1;\n"
                    "   else if (start1 < 0)\n"
                    "       start1 = max_dim1 + start1;\n"
                    "   if (start2 > (max_dim2-1))\n"
                    "       start2 = max_dim2-1;\n"
                    "   else if (start2 < 0)\n"
                    "       start2 = max_dim2 + start2;\n"
                    "   if (start3 > (max_dim3-1))\n"
                    "       start3 = max_dim3-1;\n"
                    "   else if (start3 < 0)\n"
                    "       start3 = max_dim3 + start3;\n"
                    
                    "   int index1 = start1 + x;\n"
                    "   int index2 = start2 + y;\n"
                    "   int index3 = start3 + c;\n"
                    "   index1 = (dim1 >= 0) ? index1 : 0;\n"
                    "   index2 = (dim2 >= 0) ? index2 : 0;\n"
                    "   index3 = (dim3 >= 0) ? index3 : 0;\n"
                    "   half value;\n"
                    "   uint offset;\n"
                    "   value = *(__global half*)&in[in_offset + index1*in_stride[0] + index2*in_stride[1] + index3*in_stride[2]];\n"
                    "   offset = out_offset + x*out_stride[0] + y*out_stride[1] + c*out_stride[2];\n"
                    "   out += offset;\n"
                    "   *(__global half *)&out[0] = value;\n"
                    "}\n", (int)in_stride[0], (int)in_stride[1], (int)in_stride[2], (int)in_stride[3], (int)out_stride[0], (int)out_stride[1], (int)out_stride[2], (int)out_stride[3], (int)starts_stride[0], data->in_offset, data->out_offset, data->starts_offset, (int)num_of_dims, max_dim[0], max_dim[1], max_dim[2]);
            }
        }
    }

    // build OpenCL C code and save the kernel object
    cl_context opencl_context = nullptr;
    cl_device_id device_id = nullptr;
    ERROR_CHECK_STATUS(clGetCommandQueueInfo(data->handle->cmdq, CL_QUEUE_CONTEXT, sizeof(cl_context), &opencl_context, nullptr));
    ERROR_CHECK_STATUS(clGetCommandQueueInfo(data->handle->cmdq, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr));
    const char * program_src[] = { item };
    cl_int err;
    cl_program program = clCreateProgramWithSource(opencl_context, 1, program_src, nullptr, &err);
    if(!program) {
        printf("ERROR: clCreateProgramWithSource failed with (%d) for below code:\n<<<\n%s\n>>>\n", err, item);
        return VX_FAILURE;
    }
    err = clBuildProgram(program, 1, &device_id, "", nullptr, nullptr);
    if(err) {
        printf("ERROR: clBuildProgram failed with (%d) for below code:\n<<<\n%s\n>>>\n", err, item);
        return VX_FAILURE;
    }
    data->slice_kernel = clCreateKernel(program, "slice", &err);
    if(!data->slice_kernel) {
        printf("ERROR: Slice: clCreateKernel(*,slice,*) failed with (%d)\n", err);
        return VX_FAILURE;
    }
    ERROR_CHECK_STATUS(clReleaseProgram(program));

    //reverse input dims w,h,c,n- > n,c,h,w
    int start = 0, end = num_of_dims-1;
    while (start < end) {
        int temp = input_dims[start];
        input_dims[start] = input_dims[end];
        input_dims[end] = temp;
        start++;
        end--;
    }

    // execute slice kernel first time
    ERROR_CHECK_STATUS(clSetKernelArg(data->slice_kernel, 0, sizeof(cl_mem), &input_mem));
    ERROR_CHECK_STATUS(clSetKernelArg(data->slice_kernel, 1, sizeof(cl_mem), &output_mem));
    ERROR_CHECK_STATUS(clSetKernelArg(data->slice_kernel, 2, sizeof(cl_mem), &starts_mem));
    ERROR_CHECK_STATUS(clSetKernelArg(data->slice_kernel, 3, sizeof(cl_mem), &ends_mem));
    if(parameters[4]){
        ERROR_CHECK_STATUS(clSetKernelArg(data->slice_kernel, 4, sizeof(cl_mem), &axes_mem));
    }
    if(parameters[5]){
        ERROR_CHECK_STATUS(clSetKernelArg(data->slice_kernel, 5, sizeof(cl_mem), &steps_mem));
    }

    vx_status status;
    vx_map_id map_id;
    vx_size stride[4];
    int * ptr;

    int starts_value[element_count], ends_value[element_count], steps_value[element_count];

    // copy starts tensors
    status = vxMapTensorPatch((vx_tensor)parameters[2], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for starts tensor (" << status << ")" << std::endl;
        return -1;
    }
    
    for(int i=0; i<element_count; i++) {
        starts_value[i] = (int)ptr[i];
        if(starts_value[i] < 0) {
            starts_value[i] = input_dims[i] + starts_value[i];
        }
        else if(starts_value[i] > input_dims[i]) {
            starts_value[i] = input_dims[i];
        }
    }
    vxUnmapTensorPatch((vx_tensor)parameters[2], map_id);
    
    // copy ends tensors
    status = vxMapTensorPatch((vx_tensor)parameters[3], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for ends tensor (" << status << ")" << std::endl;
        return -1;
    }
    for(int i=0; i<element_count; i++) {
        ends_value[i] = (int)ptr[i];
        if(ends_value[i] < 0) {
            ends_value[i] = input_dims[i] + ends_value[i];
        }
        else if(ends_value[i] > input_dims[i]) {
            ends_value[i] = input_dims[i];
        }
    }
    vxUnmapTensorPatch((vx_tensor)parameters[3], map_id);
    
    if (parameters[5]) {
        // copy steps tensors
        status = vxMapTensorPatch((vx_tensor)parameters[5], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if(status) {
            std::cerr << "ERROR: vxMapTensorPatch() failed for steps tensor (" << status << ")" << std::endl;
            return -1;
        }
        
        for(int i=0; i<element_count; i++) {
            steps_value[i] = (int)ptr[i];
            if(steps_value[i] == 0) {
                std::cerr << "ERROR: steps value can't be zero" << std::endl;
                return -1;
            }
        }
        vxUnmapTensorPatch((vx_tensor)parameters[5], map_id);
    }
    else {
        for(int i=0; i<element_count; i++) {
            steps_value[i] = 1;
        }
    }
    // calculate global dims
    size_t slice_global[3];
    slice_global[0] = 1;
    slice_global[1] = 1;
    slice_global[2] = 1;
    for(int i=0; i<element_count; i++) {
        int value = starts_value[element_count-i-1];
        slice_global[i] = 0;
        while(value < ends_value[element_count-i-1]) {
            slice_global[i]++;
            value+=steps_value[element_count-i-1];
        }
    }
    ERROR_CHECK_STATUS(clEnqueueNDRangeKernel(data->handle->cmdq, data->slice_kernel, 3, nullptr, slice_global, nullptr, 0, nullptr, nullptr));
    ERROR_CHECK_STATUS(clFinish(data->handle->cmdq));
    
    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status publishSliceLayer(vx_context context) 
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.slice_layer", VX_KERNEL_SLICE_LAYER_AMD, initializeSliceLayer, 4, validateSliceLayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
    
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    
    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS; 
}

VX_API_ENTRY vx_node VX_API_CALL vxSliceLayer(vx_graph graph, vx_tensor input, vx_tensor output, vx_tensor starts, vx_tensor ends, vx_tensor axes, vx_tensor steps) 
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference) input,
            (vx_reference) output,
            (vx_reference) starts,
            (vx_reference) ends,
            (vx_reference) axes,
            (vx_reference) steps
        };
        node = createNode(graph, VX_KERNEL_SLICE_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}