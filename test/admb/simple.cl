
__kernel void AD(__global struct ad_gradient_structure* gs,
        __global struct ad_entry* gradient_stack,
        __constant struct ad_variable* a,
        __constant struct ad_variable*b,
        __global double *x,
        __global double *y,
        __global struct ad_variable *out, int size) {




    //get global id
    const int id = get_global_id(0);

    if (id < size) {
        
        //declare a private gradient structure and use pad operations.    
        struct ad_gradient_structure pgs;

        //initialize the global gradient structure
        ad_init(gs, gradient_stack);
        
        //initialize the private gradient structure and prefetch our stack entries.
        pad_init(4, &pgs, gs, gradient_stack);
        
        struct ad_variable aa = *a;
        struct ad_variable bb = *b;
        double xx = x[id];
        double yy = y[id];

        struct ad_variable temp = pad_minus_vd(&pgs, pad_plus(&pgs, pad_times_vd(&pgs, aa, xx), bb), yy);
        out[id] = pad_times(&pgs, temp, temp);
    }
}
