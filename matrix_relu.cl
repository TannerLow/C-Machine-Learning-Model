/* matrix_relu.cl 
 * Matrix + Row: C = Relu(A)
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixRelu(__global float* matrix, 
           __global float* out)
{
    // thread index
    int i = get_global_id(0);

    out[i] = matrix[i] > 0 ? matrix[i] : 0;
}