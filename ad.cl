/**
 * A simple API to achieve reverse mode automatic differentiation of computer 
 * programs on a GPU using OpenCL. 
 */


#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
//#error "Double precision floating point not supported by OpenCL implementation."
#endif


#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#endif



#if defined(DOUBLE_SUPPORT_AVAILABLE)

// double
typedef double real_t;
typedef double2 real2_t;
typedef double3 real3_t;
typedef double4 real4_t;
typedef double8 real8_t;
typedef double16 real16_t;
#define PI 3.14159265358979323846

#else

// float
typedef float real_t;
typedef float2 real2_t;
typedef float3 real3_t;
typedef float4 real4_t;
typedef float8 real8_t;
typedef float16 real16_t;
#define PI 3.14159265359f

#endif

#ifndef MAX_VARIABLE_IN_EXPESSION
#define MAX_VARIABLE_IN_EXPESSION 2
#endif

//#define /*__private*/

//__local int stack_id;
//__local int current_id;

struct /*__attribute__ ((packed))*/ ad_variable {
    double value;
    int id;
};

struct /*__attribute__ ((packed))*/ ad_pair {
    double dx;
    int id;
};

struct /*__attribute__ ((aligned))*/ ad_entry {
    struct ad_pair coeff[MAX_VARIABLE_IN_EXPESSION];
    int id;
    int size;
};

struct /*__attribute__ ((packed))*/ ad_gradient_structure {
    __global struct ad_entry* gradient_stack;
    int current_ad_variable_id;
    int stack_current;
    int recording;
    int counter;
    __global struct ad_entry* index;

};

struct local_gradient_structure {
    struct ad_entry gradient_stack[4*64+1];
    int current_ad_variable_id;
    int stack_current;
    int recording;
    int counter;
    int gcounter;
};

inline void ad_init(__global struct ad_gradient_structure* gs, __global struct ad_entry * gradient_stack) {
    gs->gradient_stack = gradient_stack;
}

inline void pad_init(int operations, struct ad_gradient_structure* pgs, __global struct ad_gradient_structure* gs, __global struct ad_entry * gradient_stack) {

    if (gs->recording == 1) {
        pgs->gradient_stack = gradient_stack;
        pgs->counter = atomic_add(&gs->counter, operations);
        pgs->current_ad_variable_id = gs->current_ad_variable_id;
        pgs->stack_current = gs->stack_current;
        pgs->recording = gs->recording;
        pgs->index = &gs->gradient_stack[gs->stack_current];
    } else {
        pgs->recording = 0;
    }
}

inline void lad_init(int operations, int work_group_size,
        __local struct local_gradient_structure* lgs,
        __global struct ad_gradient_structure* gs,
        __global struct ad_entry * gradient_stack) {

    if (get_local_id(0) == 0) {
        if (gs->recording == 1) {
            lgs->counter = 0;
            lgs->gcounter = atomic_add(&gs->counter, operations * work_group_size);
            lgs->current_ad_variable_id = gs->current_ad_variable_id;
            lgs->stack_current = gs->stack_current;
            lgs->recording = 1;
        } else {
            lgs->recording = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/**
 * Adds two ad_variables together. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_plus(__global struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = a.value + b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e->coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e->size = 2;
    }

    return ret;
}

/**
 * Adds ad_variable a to double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_plus_vd(__global struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    struct ad_variable ret = {.value = a.value + b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e->size = 1;
    }

    return ret;
}

/**
 * Adds double a to ad_variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_plus_dv(__global struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a + b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Plus assign ad_variable a and ad_variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 */
inline void ad_plus_eq(__global struct ad_gradient_structure* gs, struct ad_variable* a, struct ad_variable b) {
    a->value += b.value;

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
        e->coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e->size = 2;
        e->id = a->id;
    }
}

inline void ad_plus_eq_g(__global struct ad_gradient_structure* gs, __global struct ad_variable* a, struct ad_variable b) {
    a->value += b.value;

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
        e->coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e->size = 2;
        e->id = a->id;
    }
}

/**
 * Plus assign ad_variable a and double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 *  
 * @param gs
 * @param a
 * @param b
 */
inline void ad_plus_eq_d(__global struct ad_gradient_structure* gs, struct ad_variable* a, double b) {
    a->value += b;

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
        e->size = 1;
        e->id = a->id;
    }
}

/**
 * Subtracts ad_variable b from ad_variable a. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_minus(__global struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = a.value - b.value, .id = 0};

    if (gs->recording) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e->coeff[1] = (struct ad_pair){.dx = -1.0, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Subtracts double b from ad_variable a.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_minus_vd(__global struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    struct ad_variable ret = {.value = a.value - b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e->size = 1;
    }

    return ret;
}

/**
 * Subtracts ad_variable b from double a. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 *  
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_minus_dv(__global struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a - b.value, .id = 0};

    if (gs->recording) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = -1.0, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Multiplies to ad_variables together. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_times(__global struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = a.value * b.value, .id = 0};


    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct ad_pair){.dx = a.value, .id = a.id};
        e->coeff[1] = (struct ad_pair){.dx = b.value, .id = b.id};
        e->size = 2;
    }

    return ret;
}

/**
 * Multiplies ad_variable a and double b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_times_vd(__global struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    struct ad_variable ret = {.value = a.value * b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = b, .id = a.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Multiplies double a and ad_variable b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_times_dv(__global struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a * b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = b.value, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Divides ad_variable a by ad_variable b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_divide(__global struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = a.value / b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b.value;
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
        e->coeff[1] = (struct ad_pair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Divides ad_variable a by double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed. 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_divide_vd(__global struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    struct ad_variable ret = {.value = a.value / b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b;
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
        e->size = 1;
        e->id = ret.id;
    }
    return ret;
}

/**
 * Divides double a by ad_variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable ad_divide_dv(__global struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a / b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b.value;
        e->coeff[0] = (struct ad_pair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Adds two ad_variables together. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_plus(struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    //    struct ad_variable ret = {.value = a.value + b.value, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        struct ad_variable ret = {.value = a.value + b.value, .id = gs->current_ad_variable_id + index};
        //        ret.id = index + gs->current_ad_variable_id;
        //        __global struct ad_entry* e =
        //                &gs->gradient_stack[index + gs->stack_current];
        struct ad_entry e;
        e.id = ret.id;
        e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e.coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e.size = 2;

        gs->gradient_stack[index + gs->stack_current] = e;
        return ret;
    } else {

        return (struct ad_variable) {
            .value = a.value + b.value, .id = 0
        };
    }

    //    return ret;
}

/**
 * Adds ad_variable a to double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_plus_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    struct ad_variable ret = {.value = a.value + b, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e->size = 1;
    }

    return ret;
}

/**
 * Adds double a to ad_variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_plus_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a + b.value, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Plus assign ad_variable a and ad_variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 */
inline void pad_plus_eq(struct ad_gradient_structure* gs, struct ad_variable* a, struct ad_variable b) {
    a->value += b.value;

    if (gs->recording == 1) {
        int index = gs->counter++;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
        e->coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e->size = 2;
        e->id = a->id;
    }
}

inline void pad_plus_eq_g(struct ad_gradient_structure* gs, __global struct ad_variable* a, struct ad_variable b) {
    a->value += b.value;

    if (gs->recording == 1) {
        int index = gs->counter++;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
        e->coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
        e->size = 2;
        e->id = a->id;
    }
}

/**
 * Plus assign ad_variable a and double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 *  
 * @param gs
 * @param a
 * @param b
 */
inline void pad_plus_eq_d(struct ad_gradient_structure* gs, struct ad_variable* a, double b) {
    a->value += b;

    if (gs->recording == 1) {
        int index = gs->counter++;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
        e->size = 1;
        e->id = a->id;
    }
}

/**
 * Subtracts ad_variable b from ad_variable a. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_minus(struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = a.value - b.value, .id = 0};

    if (gs->recording) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e->coeff[1] = (struct ad_pair){.dx = -1.0, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Subtracts double b from ad_variable a.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_minus_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    struct ad_variable ret = {.value = a.value - b, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        //        __global struct ad_entry* e =
        //                &gs->gradient_stack[index + gs->stack_current];
        struct ad_entry e;
        e.id = ret.id;
        e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
        e.size = 1;
        gs->gradient_stack[index + gs->stack_current] = e;
    }

    return ret;
}

/**
 * Subtracts ad_variable b from double a. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 *  
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_minus_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a - b.value, .id = 0};

    if (gs->recording) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = -1.0, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Multiplies to ad_variables together. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_times(struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = a.value * b.value, .id = 0};


    if (gs->recording == 1) {
        int index = gs->counter++;
//        ret.id = index + gs->current_ad_variable_id;
          struct ad_variable ret = {.value = a.value * b.value, .id = index + gs->current_ad_variable_id};
        //        __global struct ad_entry* e =
        //                &gs->gradient_stack[index + gs->stack_current];
        struct ad_entry e;
//        (struct ad_entry){.coeff ={{.dx = a.value, .id = a.id},{.dx = b.value, .id = b.id}}, .id=ret.id, .size=2};
        e.id = ret.id;
//////          struct ad_pair data[] ={{.dx = a.value, .id = a.id},{.dx = b.value, .id = b.id}};
//////        e.coeff = data;
        e.coeff[0] = (struct ad_pair){.dx = a.value, .id = a.id};
        e.coeff[1] = (struct ad_pair){.dx = b.value, .id = b.id};
        e.size = 2;
        gs->gradient_stack[index + gs->stack_current] = e;
        
        return ret;
    }else{
        return (struct ad_variable) {.value = a.value * b.value, .id = 0};
    }

//    return ret;
}

/**
 * Multiplies ad_variable a and double b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_times_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    //    struct ad_variable ret = {.value = a.value * b, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        struct ad_variable ret = {.value = a.value * b, .id = index + gs->current_ad_variable_id};

        //        ret.id = index + gs->current_ad_variable_id;
        //        __global struct ad_entry* e =
        //                &gs->gradient_stack[index + gs->stack_current];
        struct ad_entry e;
        e.coeff[0] = (struct ad_pair){.dx = b, .id = a.id};
        e.size = 1;
        e.id = ret.id;
        gs->gradient_stack[index + gs->stack_current] = e;
        return ret;
    } else {

        return (struct ad_variable) {
            .value = a.value * b, .id = 0
        };
    }

    //    return ret;
}

/**
 * Multiplies double a and ad_variable b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_times_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a * b.value, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = a, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Divides ad_variable a by ad_variable b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_divide(struct ad_gradient_structure* gs, const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = a.value / b.value, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b.value;
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
        e->coeff[1] = (struct ad_pair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Divides ad_variable a by double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed. 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_divide_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
    struct ad_variable ret = {.value = a.value / b, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        //        __global struct ad_entry* e =
        //                &gs->gradient_stack[index + gs->stack_current];
        struct ad_entry e;
        double inv = 1.0 / b;
        e.coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
        e.size = 1;
        e.id = ret.id;
        gs->gradient_stack[index + gs->stack_current] = e;
    }
    return ret;
}

/**
 * Divides double a by ad_variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct ad_variable pad_divide_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
    struct ad_variable ret = {.value = a / b.value, .id = 0};

    if (gs->recording == 1) {
        int index = gs->counter++;
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b.value;
        e->coeff[0] = (struct ad_pair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_cos(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = log(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = -1.0 * sin(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_sin(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = sin(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = cos(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_tan(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = tan(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = 1.0 / cos(v.value);
        e->coeff[0] = (struct ad_pair){.dx = temp*temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_acos(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = acos(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (-1.0) /
                pow(((1.0) -
                pow(v.value, (2.0))),
                (0.5));
        e->coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_asin(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = asin(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (1.0) /
                pow(((1.0) -
                pow(v.value, (2.0))),
                (0.5));
        e->coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_atan(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = atan(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (1.0) / (v.value * v.value + (1.0));
        e->coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_cosh(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = cosh(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = sinh(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_sinh(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = sinh(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = cosh(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_tanh(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = tanh(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (1.0 / cosh(v.value))*(1.0 / cosh(v.value));
        e->coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_exp(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = exp(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct ad_pair){.dx = ret.value, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_log(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = log(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / v.value;
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_log10(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = log10(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / (v.value * 2.30258509299404590109361379290930926799774169921875);
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_pow(__global struct ad_gradient_structure* gs,
        const struct ad_variable a, const struct ad_variable b) {
    struct ad_variable ret = {.value = pow(a.value, b.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = b.value * pow(a.value, b.value - (1.0));
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
        e->coeff[1] = (struct ad_pair){.dx = log(a.value) * ret.value, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_pow_vd(__global struct ad_gradient_structure* gs,
        struct ad_variable a, double b) {
    struct ad_variable ret = {.value = pow(a.value, b), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = b * pow(a.value, b - (1.0));
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
        e->size = 1;
        e->id = ret.id;
    }
    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_pow_dv(__global struct ad_gradient_structure* gs,
        double a, struct ad_variable b) {
    struct ad_variable ret = {.value = pow(a, b.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = b.value * pow(a, b.value - (1.0));
        e->coeff[0] = (struct ad_pair){.dx = log(a) * ret.value, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct ad_variable __attribute__((overloadable)) ad_sqrt(__global struct ad_gradient_structure* gs, struct ad_variable v) {
    struct ad_variable ret = {.value = sqrt(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_ad_variable_id;
        __global struct ad_entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = .5 / ret.value;
        e->coeff[0] = (struct ad_pair){.dx = inv, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

//
