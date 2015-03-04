#define GLOBAL

#ifdef GLOBAL

__kernel void matrixMult(__global struct ad_gradient_structure* gs,
        __global struct ad_entry* gradient_stack,
        __global struct ad_variable* A,
        __global struct ad_variable* B,
        __global struct ad_variable* C,
        int widthA,
        int widthB) {


    //initialize the global gradient structure
    ad_init(gs, gradient_stack);

    int i = get_global_id(0);
    int j = get_global_id(1);

    struct ad_variable value;
    ad_init_var_g(gs, &value, 0.0);

#pragma unroll
    for (int k = 0; k < widthA; k++) {
        ad_plus_eq(gs, &value, ad_times(gs, A[k + j * widthA], B[k * widthB + i]));
    }
    C[i + widthA * j] = value;
}
#elif defined(USE_LOCAL)

__kernel void matrixMult(__global struct ad_gradient_structure* gs,
        __global struct ad_entry* gradient_stack,
        __global struct ad_variable* A,
        __global struct ad_variable* B,
        __global struct ad_variable* C,
        int widthA,
        int widthB) {

    //initialize the global gradient structure
    struct ad_gradient_structure pgs;

    //initialize the global gradient structure
    ad_init(gs, gradient_stack);

    //initialize the private gradient structure and prefetch our stack entries.
    pad_init(widthA * 2, &pgs, gs, gradient_stack);

    int i = get_global_id(0);
    int j = get_global_id(1);

    struct ad_variable value;
    ad_init_var_g(gs, &value, 0.0);

    for (int k = 0; k < widthA; k++) {

        struct ad_variable a = A[k + j * widthA];
        struct ad_variable b = B[k * widthB + i];
        pad_plus_eq(&pgs, &value, pad_times(&pgs, a, b));
    }
    C[i + widthA * j] = value;
}

#else

/* Matrix multiplication: C = A * B.
 * Device code.
 */

// Thread block size
#define BLOCK_SIZE 16

//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//////////////////////////////////////////////////////

__kernel void matrixMult(__global struct ad_gradient_structure* gs,
        __global struct ad_entry* gradient_stack,
        __global struct ad_variable* A,
        __global struct ad_variable* B,
        __global struct ad_variable* C,
        int wA, int wB) {

    //initialize the global gradient structure
    ad_init(gs, gradient_stack);


    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep = BLOCK_SIZE * wB;
    
    struct ad_variable value;
    ad_init_var_g(gs, &value, 0.0);

    // Declaration of the local memory array As 
    // used to store the sub-matrix of A
    __local struct ad_variable As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the local memory array Bs 
    // used to store the sub-matrix of B
    __local struct ad_variable Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {



        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix


#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            //            struct ad_variable a = As[ty][k];
            //            struct ad_variable b = Bs[k][tx];
            //            value = ad_plus(gs,a,b);
            ////            ad_plus_eq(gs, &value,a);
            ////            ad_times(gs, a, b);
            ad_plus_eq(gs, &value, ad_times(gs, As[ty][k], Bs[k][tx]));
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = value;

}

#endif

