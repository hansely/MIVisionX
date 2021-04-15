#include <kernels.h>

struct LocalData {
    NeuralNetworkCommonHandle * handle;
    int in_offset = 0, ldi;
    int out_offset = 0, ldo;
    int starts_offset = 0, lds;
    int ends_offset = 0, lde;
    int axes_offset = 0, lda;
    int steps_offset = 0, ldst;
    int ID;
    cl_kernel slice_kernel;
    size_t slice_global[3];
    size_t slice_local[3];
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
    
    // if (parameters[4]) {
    //     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, &axes_dims, sizeof(axes_dims)));   
    //     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &axes_type, sizeof(axes_type)));
    //     if ((axes_type != VX_TYPE_INT32) && (axes_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    //     if (starts_dims != axes_dims) {
    //         printf("validate:slice: The dimension length of starts, ends, axes, and steps must be the same.\n");
    //         return VX_ERROR_INVALID_DIMENSION;
    //     }
    // }
    
    // if (parameters[5]) {
    //     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, &steps_dims, sizeof(steps_dims)));   
    //     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &steps_type, sizeof(steps_type)));
    //     if ((steps_type != VX_TYPE_INT32) && (steps_type != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    //     if (starts_dims != steps_dims) {
    //         printf("validate:slice: The dimension length of starts, ends, axes, and steps must be the same.\n");
    //         return VX_ERROR_INVALID_DIMENSION;
    //     }
    // }

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

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &params_dim, sizeof(params_dim)));
    vx_size params_dims[params_dim];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, params_dims, sizeof(params_dims)));

    int element_count = 1;

    for (int i=0; i<params_dim; i++)
        element_count *= params_dims[i];
    
    printf("elemen count %d \n", element_count);
    // vx_size ends_dims[params_dim];
    // ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, ends_dims, sizeof(ends_dims)));
    
    // vx_size steps_dims[params_dim];
    // ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, steps_dims, sizeof(steps_dims)));
    
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
    printf("in %d %d %d %d %d out %d %d %d %d %d start %d steps %d\n", in_stride[0], in_stride[1], in_stride[2], in_stride[3], data->in_offset, out_stride[0], out_stride[1], out_stride[2], out_stride[3], data->out_offset, (int)data->starts_offset, (int)data->steps_offset);
    printf("starts %d %d %d %d steps %d %d %d %d \n", starts_stride[0], starts_stride[1], starts_stride[2], starts_stride[3], steps_stride[0], steps_stride[1], steps_stride[2], steps_stride[3]);
    if (num_of_dims) {
        if (type == VX_TYPE_FLOAT32) {
            printf("opencl gen %d\n", (int)num_of_dims);
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
                "   printf(\"x: %%d y: %%d c: %%d 1: %%d 2: %%d 3: %%d value %%f\\n\", x, y, c, index1, index2, index3, value); \n"
                "   offset = out_offset + x*out_stride[0] + y*out_stride[1] + c*out_stride[2];\n"
                "   out += offset;\n"
                "   *(__global float *)&out[0] = value;\n"
                "}\n", (int)in_stride[0], (int)in_stride[1], (int)in_stride[2], (int)in_stride[3], (int)out_stride[0], (int)out_stride[1], (int)out_stride[2], (int)out_stride[3], (int)starts_stride[0], (int)steps_stride[0], data->in_offset, data->out_offset, data->starts_offset, data->steps_offset, (int)num_of_dims, max_dim[0], max_dim[1], max_dim[2]);
        }
        // else {
        //     sprintf(item,
        //         "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
        //         "__kernel void slice_layer(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * ind, uint ind_offset, uint4 ind_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint axis) \n"
        //         "{ \n"
        //         "   uint x = get_global_id(0);\n"
        //         "   uint y = get_global_id(1);\n"
        //         "   uint c = get_global_id(2);\n"
        //         "   int indices = *(__global int*)&ind[ind_offset + y*ind_stride.s0];\n"
        //         "   half value;\n"
        //         "   uint offset;\n"
        //         "   if (axis == 0) {\n"
        //         "       value = *(__global half*)&in[in_offset + x*in_stride.s0 + indices*in_stride.s1 + c*in_stride.s2];\n"
        //         "       offset = out_offset + x*out_stride.s0 + y*out_stride.s1 + c*out_stride.s2;\n"
        //         "   }\n"
        //         "   else if (axis == 1) {\n"
        //         "       value = *(__global half*)&in[in_offset + indices*in_stride.s0 + c*in_stride.s1];\n"
        //         "       offset = out_offset + y*out_stride.s0 + c*out_stride.s1;\n"
        //         "   }\n"
        //         "   else if (axis == 2) {\n"
        //         "       value = *(__global half*)&in[in_offset + c*in_stride.s0];\n"
        //         "       offset = out_offset + c*out_stride.s0;\n"
        //         "   }\n"
        //         "   out += offset;\n"
        //         "   *(__global half *)&out[0] = value;\n"
        //         "}\n");
        // }
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

    // copy input tensors
    status = vxMapTensorPatch((vx_tensor)parameters[2], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for starts tensor (" << status << ")" << std::endl;
        return -1;
    }
    
    for(int i=0; i<element_count; i++) {
        starts_value[element_count] = (int)ptr[i];
        printf("the starts value[%d] : %d\n", i, (int)ptr[i]);
    }
    vxUnmapTensorPatch((vx_tensor)parameters[2], map_id);
    
    status = vxMapTensorPatch((vx_tensor)parameters[3], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for ends tensor (" << status << ")" << std::endl;
        return -1;
    }
    
    for(int i=0; i<element_count; i++) {
        ends_value[element_count] = (int)ptr[i];
        printf("the ends value[%d] : %d\n", i, (int)ptr[i]);
    }
    vxUnmapTensorPatch((vx_tensor)parameters[3], map_id);
    
    status = vxMapTensorPatch((vx_tensor)parameters[4], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for steps tensor (" << status << ")" << std::endl;
        return -1;
    }
    
    for(int i=0; i<element_count; i++) {
        steps_value[element_count] = (int)ptr[i];
        printf("the steps value[%d] : %d\n", i, (int)ptr[i]);
    }
    vxUnmapTensorPatch((vx_tensor)parameters[4], map_id);
    
    size_t slice_global[3];
    slice_global[0] = 2;
    slice_global[1] = 1;
    slice_global[2] = 1;
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


// static vx_status VX_CALLBACK processSliceLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
// {
//     //get input params
//     vx_status status;
//     vx_enum input_type, type;
//     vx_map_id map_id;
//     vx_size stride[4];
//     vx_size icount = 1, ocount = 1;
//     std::vector<float> input, output;
//     std::vector<int> starts, ends, axes, steps;
//     float * fptr;
//     int * ptr;

//     vx_size input_num_dims, input_dims[4], param_dims, output_num_dims, output_dims[4];
//     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &input_num_dims, sizeof(input_num_dims)));
//     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
//     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));

//     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &output_num_dims, sizeof(output_num_dims)));
//     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

//     for(int i=0; i<input_num_dims; i++) {
//         icount *= input_dims[i];
//     }

//     for(int i=0; i<output_num_dims; i++) {
//         ocount *= output_dims[i];
//     }

//     // check the number of param dimensions and type
//     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &param_dims, sizeof(param_dims)));   
//     ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

//     // create array of input/indices/output tensors
//     vx_tensor input_tensor[param_dims], indices_tensor[param_dims], output_tensor[param_dims];
    
//     // create context & graph
//     vx_context context = vxCreateContext();
//     status = vxGetStatus((vx_reference)context);
//     if(status) {
//         std::cerr << "ERROR: vxCreateContext() failed (" << status << ")" << std::endl;
//         return -1;
//     }

//     vxLoadKernels(context, "vx_nn"); // load vx_nn kernels

//     vx_graph graph = vxCreateGraph(context);
//     status = vxGetStatus((vx_reference)graph);
//     if(status) {
//         std::cerr << "ERROR: vxCreateGraph() failed (" << status << ")" << std::endl;
//         return -1;
//     }

//     // copy input tensor
//     status = vxMapTensorPatch((vx_tensor)parameters[0], 2, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
//     if(status) {
//         std::cerr << "ERROR: vxMapTensorPatch() failed for input tensor (" << status << ")" << std::endl;
//         return -1;
//     }
//     for(int i=0; i<8; i++) {
//         input.push_back((float)fptr[i]);
//     }
//     vxUnmapTensorPatch((vx_tensor)parameters[0], map_id);

//     // create tmp_input and copy input tensor
//     vx_tensor tmpi_tensor = vxCreateTensor(context, input_num_dims, input_dims, input_type, 0);
//     status = vxMapTensorPatch(tmpi_tensor, input_num_dims, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
//     if(status) {
//         std::cerr << "ERROR: vxMapTensorPatch() failed for tmpi tensor (" << status << ")" << std::endl;
//         return -1;
//     }
//     for(int i=0; i<icount; i++) {
//         fptr[i] = input[i];
//     }
//     vxUnmapTensorPatch(tmpi_tensor, map_id);

//     input_tensor[0]  = tmpi_tensor;

//     // copy starts index
//     status = vxMapTensorPatch((vx_tensor)parameters[2], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
//     if(status) {
//         std::cerr << "ERROR: vxMapTensorPatch() failed for starts tensor (" << status << ")" << std::endl;
//         return -1;
//     }

//     for(int i=0; i<param_dims; i++) {
//         int num_element = input_dims[input_num_dims-1-i];
//         int value = (int)ptr[i];
//         if (value >= num_element) // If the value passed to starts is larger than the n(the number of elements in this dimension), it represents n
//             starts.push_back(num_element);
//         else if (value < 0) {
//             starts.push_back(num_element+value); // If a negative value is passed for any of the starts indices, it represents number of elements before the end of that dimension
//         }
//         else
//             starts.push_back(value);
//     }

//     vxUnmapTensorPatch((vx_tensor)parameters[2], map_id);

//     // copy ends index
//     status = vxMapTensorPatch((vx_tensor)parameters[3], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
//     if(status) {
//         std::cerr << "ERROR: vxMapTensorPatch() failed for ends tensor (" << status << ")" << std::endl;
//         return -1;
//     }

    
//     for(int i=0; i<param_dims; i++) {
//         int num_element = input_dims[input_num_dims-1-i];
//         int value = (int)ptr[i];
//         if (value >= num_element) // If the value passed to ends is larger than the n(the number of elements in this dimension), it represents n
//             ends.push_back(num_element);
//         else if (value < 0) {
//             ends.push_back(num_element+value); // If a negative value is passed for any of the ends indices, it represents number of elements before the end of that dimension
//         }
//         else
//             ends.push_back(value);
//     }


//     vxUnmapTensorPatch((vx_tensor)parameters[3], map_id);

//     // copy axes index        
//     if(parameters[4]) {
//         status = vxMapTensorPatch((vx_tensor)parameters[4], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
//         if(status) {
//             std::cerr << "ERROR: vxMapTensorPatch() failed for axes tensor (" << status << ")" << std::endl;
//             return -1;
//         }

//         for(int i=0; i<param_dims; i++) {
//             axes.push_back((int)ptr[i]);
//         }

//         vxUnmapTensorPatch((vx_tensor)parameters[4], map_id);
//     }
//     else {
//         for(int i=0; i<param_dims; i++)
//             axes.push_back(i);
//     }

//     // copy steps index
//     if(parameters[5]) {
//         status = vxMapTensorPatch((vx_tensor)parameters[5], 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
//         if(status) {
//             std::cerr << "ERROR: vxMapTensorPatch() failed for steps tensor (" << status << ")" << std::endl;
//             return -1;
//         }

//         for(int i=0; i<param_dims; i++) {
//             steps.push_back((int)ptr[i]);
//         }
//         vxUnmapTensorPatch((vx_tensor)parameters[5], map_id);
//     }
//     else {
//         for(int i=0; i<param_dims; i++)
//             steps.push_back(1);
//     }

//     // calculate and create indices tensor
//     std::vector<std::vector<int>> indices(param_dims);
//     for(int i=0; i<param_dims; i++) {
//         int index = starts[i];
//         while (index < ends[i]) {
//             indices[i].push_back(index);
//             index += steps[i];
//         }

//         vx_size dim[1] = {indices[i].size()};
        
//         vx_tensor tmp_tensor = vxCreateTensor(context, 1, dim, type, 0);
//         status = vxMapTensorPatch(tmp_tensor, 1, nullptr, nullptr, &map_id, stride, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
//         if(status) {
//             std::cerr << "ERROR: vxMapTensorPatch() failed for indices tensor(" << status << ")" << std::endl;
//             return -1;
//         }
        
//         for(auto itr = indices[i].begin(); itr != indices[i].end(); itr++) {
//             *ptr++ = *itr;
//         }
//         vxUnmapTensorPatch(tmp_tensor, map_id);
//         indices_tensor[i] = tmp_tensor;
//     }

//     // calculate output dims and create output tensor
//     vx_size tmp_num_dims = input_num_dims;
//     vx_size *tmp_dims = input_dims;

//     for(int i=0; i<(param_dims-1); i++) {
//         //reverse input dims w,h,c,n- > n,c,h,w
//         int start = 0, end = tmp_num_dims-1;
//         while (start < end) {
//             int temp = tmp_dims[start];
//             tmp_dims[start] = tmp_dims[end];
//             tmp_dims[end] = temp;
//             start++;
//             end--;
//         }

//         int out_dim_rank = tmp_num_dims;    
//         vx_size out_dims[out_dim_rank];
        
//         for (int j=0; j<out_dim_rank; j++) {
//             if (j == axes[i]) {
//                 out_dims[j] = indices[i].size();
//             }
//             else {
//                 out_dims[j] = tmp_dims[j];
//             }
//         }

//         //reverse output dims n,c,h,w -> w,h,c,n
//         start = 0, end = out_dim_rank-1;
//         while (start < end) {
//             int temp = out_dims[start];
//             out_dims[start] = out_dims[end];
//             out_dims[end] = temp;
//             start++;
//             end--;
//         }
        
//         //vx_tensor tmp_tensor = vxCreateTensor(context, out_dim_rank, out_dims, input_type, 0);
//         vx_tensor tmp_tensor = vxCreateVirtualTensor(graph, out_dim_rank, out_dims, input_type, 0);
//         input_tensor[i+1] = tmp_tensor;
//         output_tensor[i] = tmp_tensor;
        
//         tmp_num_dims = out_dim_rank;
//         tmp_dims = out_dims;
//     }
    
//     // create temp output tensor which will hold the final output result
//     vx_tensor tmpo_tensor = vxCreateTensor(context, output_num_dims, output_dims, input_type, 0);
//     output_tensor[param_dims-1] = tmpo_tensor;

//     for (int i=0; i<param_dims; i++) {
//         vxGatherLayer(graph, (vx_tensor)input_tensor[i], (vx_tensor)indices_tensor[i], (vx_tensor)output_tensor[i], axes[i]);
//     }
    
//     status = vxVerifyGraph(graph);
//     if(status) {
//         std::cerr << "ERROR: vxVerifyGraph() failed (" << status << ")" << std::endl;
//         return -1;
//     }

//     status = vxProcessGraph(graph);
//     if(status) {
//         std::cerr << "ERROR: vxProcessGraph() failed (" << status << ")" << std::endl;
//         return -1;
//     }

//     // copy final output
//     status = vxMapTensorPatch(tmpo_tensor, output_num_dims, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
//     if(status) {
//         std::cerr << "ERROR: vxMapTensorPatch() failed for temp output tensor (" << status << ")" << std::endl;
//         return -1;
//     }
    
//     for(int i=0; i<ocount; i++) {
//         output.push_back((float)fptr[i]);
//     }
//     vxUnmapTensorPatch(tmpo_tensor, map_id);
 
//     // write final output
//     status = vxMapTensorPatch((vx_tensor)parameters[1], 2, nullptr, nullptr, &map_id, stride, (void **)&fptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
//     if(status) {
//         std::cerr << "ERROR: vxMapTensorPatch() failed for input3 tensor (" << status << ")" << std::endl;
//         return -1;
//     }

//     for(int i=0; i<ocount; i++) {
//         fptr[i] = output[i];
//     }
//     vxUnmapTensorPatch((vx_tensor)parameters[1], map_id);

//     return VX_SUCCESS;
// }
