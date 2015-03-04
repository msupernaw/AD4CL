
inline void compute_gradient(__global struct ad_gradient_structure* gs, __global double* gradient, int* size) {


    if (gs->recording == 1) {
        *size = gs->counter + gs->current_ad_variable_id + 1;

        for (int i = 0; i < *size; i++) {
            gradient[i] = 0.0;
        }

        gradient[gs->gradient_stack[gs->counter + gs->stack_current - 1].id] = 1.0;

#pragma unroll
        for (int j = gs->counter + gs->stack_current - 1; j >= 0; j--) {
            int id = gs->gradient_stack[j].id;
            double w = gradient[id];
//            //                if (w != 0.0) {
            gradient[id] = 0.0;
            struct ad_entry ge = gs->gradient_stack[j];
            for (int i = 0; i < ge.size; i++) {
                gradient[ge.coeff[i].id] += w * ge.coeff[i].dx;
            }
            gs->gradient_stack[j].size = 0;
            gs->gradient_stack[j].id = 0;
            //                }
        }
    }
}
//#define DOALL_ON_GPU

#ifdef DOALL_ON_GPU

__kernel void AD(__global struct ad_gradient_structure* gs,
        __global struct ad_entry* gradient_stack,
        __constant struct ad_variable* a,
        __constant struct ad_variable*b,
        __global double *x,
        __global double *y,
        __global struct ad_variable *out,
        int size,
        __global double* gradient_buffer,
        __global double* da,
        __global double* db,
        __global double* f,
        __global volatile int* counter) {




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

        atomic_dec(counter);
        //barrier(CLK_LOCAL_MEM_FENCE);

        if (id == (size-1)) {

            while(*counter > 0){}
            
            struct ad_variable sum = {.value = 0.0, .id = gs->current_ad_variable_id + atomic_inc(&gs->counter)};
            for (int i = 0; i < size; i++) {
                ad_plus_eq(gs, &sum, out[i]);
                out[i].value = 0;
            }


            struct ad_variable ff = ad_times_dv(gs, (double) (size) / 2.0, ad_log(gs, sum));
            int gsize = 0;
            compute_gradient(gs, gradient_buffer, &gsize);
            *f = gs->counter; //sum.id;//value;
            *da = gradient_buffer[a->id];
            *db = gradient_buffer[b->id];

        }

    }




}


#else

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

    //    barrier(CLK_GLOBAL_MEM_FENCE);
}
#endif