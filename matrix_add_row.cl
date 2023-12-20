/* matrix_add_row.cl 
 * Matrix + Row: C = A[M x N] + B[1 x N].
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixAddRow(__global float* matrix, 
             __global float* row, 
             __global float* out, 
             int columns)
{
    // thread index
    int rowIndex = get_global_id(0);
    
    int matrixIndex;
    for (int col = 0; col < columns; ++col) {
        matrixIndex = rowIndex * columns + col;
        out[matrixIndex] = matrix[matrixIndex] + row[col];
    }
}