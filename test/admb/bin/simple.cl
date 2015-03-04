
void compute_gradient(__global struct ad_gradient_structure* gs,__global double* gradient,  int* size) {


        if (gs->recording == 1) {
            *size = gs->current_ad_variable_id + 1;
                        for (int i = 0; i < *size; i++) {
                            gradient[i] = 0;
                        }
            gradient[gs->gradient_stack[gs->stack_current - 1].id] = 1.0;

            for (int j = gs->stack_current - 1; j >= 0; j--) {
                int id = gs->gradient_stack[j].id;
                double w = gradient[id];
//                if (w != 0.0) {
                    gradient[id] = 0.0;
                    for (int i = 0; i < gs->gradient_stack[j].size; i++) {
                        gradient[gs->gradient_stack[j].coeff[i].id] += w * gs->gradient_stack[j].coeff[i].dx;
                    }
//                }
            }
        }
    }
#define DOALL_ON_GPU

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
        __global double* f) {




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

        barrier(CLK_GLOBAL_MEM_FENCE);
        
        if (id == size-1) {

            ad_variable sum = {.value = 0.0, .id = gs->current_variable_id+gs->counter++};
            for (int i = 0; i < size; i++) {
                ad_plus_eq_v(gs, &sum, out[i]);
                out[i].value = 0;
            }


            //finish up with the native api.
            struct ad_variable ff = ad_times_dv(gs, static_cast<double> (size) / 2.0, ad_log(gs, sum));
            int gsize =0;
            compute_gradient(gs,gradient_buffer, &gsize);
            *f = ff.value;
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