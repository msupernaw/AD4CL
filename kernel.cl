
__kernel void AD(__global struct gradient_structure* gs,
        __global struct entry* gradient_stack,
        __global struct variable* a,
        __global struct variable*b,
        __global double *x,
        __global double *y,
        __global struct variable *out, int size) {

    int id = get_global_id(0);
    int lid = get_local_id(0);
  
//    if (id == 0) {
        init(gs, gradient_stack);
       
     
//    }
// barrier(CLK_GLOBAL_MEM_FENCE);

    if (id < size) {
        struct variable temp = minus_vd(gs, plus(gs, times_vd(gs, *a, x[id]), *b), y[id]);
        struct variable v = times(gs, temp, temp); // minus_vd(gs, plus_vv(gs, times_vd(gs, *a, x[i]), *b), y[i]), minus_vd(gs, plus_vv(gs, times_vd(gs, *a, x[i]), *b), y[i]));
        out[id] = v;
    }


}
