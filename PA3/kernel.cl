__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Compute C = A^T B 
   int dest_i = get_global_id(0), dest_j = get_global_id(1);
    int sum = 0;
    for (int x = 0; x < numARows; x++) {
        int a = A[x * numAColumns + dest_i];
        int b = B[x * numBColumns + dest_j];
        printf("%d * %d -> %d %d", a, b, dest_i, dest_j);
        sum += A[x * numAColumns + dest_i] * B[x * numBColumns + dest_j];
    }
    C[dest_i * numCColumns + dest_j] = sum;
}
