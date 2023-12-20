/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMultiply(__global float* C, 
          __global float* A, 
          __global float* B, 
          int sharedDimension, int bCols)
{
  
   int globalRow = get_global_id(0); 
   int globalCol = get_global_id(1);
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < sharedDimension; ++k)
   {
      float elementA = A[globalRow * sharedDimension + k];
      float elementB = B[k * bCols + globalCol];
      value += elementA * elementB;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[globalRow * bCols + globalCol] = value;
}