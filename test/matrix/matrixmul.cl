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
#else

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

#endif