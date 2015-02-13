

__kernel void AD(__global struct ad_gradient_structure* gs,
        __global struct ad_entry* gradient_stack,
        __constant  struct ad_variable* a,
        __constant  struct ad_variable*b,
        __constant double *x,
        __constant double *y,
        __global struct ad_variable *out, int size) {

    struct ad_gradient_structure pgs;

    ad_init(gs,gradient_stack);
    //initialize the gradient structure
    pad_init(4, &pgs,gs, gradient_stack);

    //get global id
   const int id = get_global_id(0);

     if (id < size) {
        struct ad_variable aa = *a;
        struct ad_variable bb = *b;
        double xx = x[id];
        double yy = y[id];
        
        struct ad_variable temp =  pad_minus_vd(&pgs, pad_plus(&pgs, pad_times_vd(&pgs, aa, xx), bb), yy);
         out[id]= pad_times(&pgs, temp, temp);
//           struct variable temp = ad_minus_vd(gs, ad_plus(gs, ad_times_vd(gs, aa, xx), bb), yy);
//        out[id]=t;//ad_times(gs, temp, temp);
    }
    

  
}