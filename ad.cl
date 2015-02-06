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

struct /*__attribute__ ((packed))*/ variable {
    double value;
    int id;
};

struct /*__attribute__ ((packed))*/ adpair {
    double dx;
    int id;
};

struct /*__attribute__ ((packed))*/ entry {
    struct adpair coeff[MAX_VARIABLE_IN_EXPESSION];
    int id;
    int size;
};

struct /*__attribute__ ((packed))*/ gradient_structure {
    __global struct entry* gradient_stack;
    int current_variable_id;
    int stack_current;
    int recording;
    int counter;

};

inline void ad_init(__global struct gradient_structure* gs, __global struct entry * gradient_stack) {
    gs->gradient_stack = gradient_stack;
}


/**
 * Adds two variables together. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_plus(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value + b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
        e->coeff[1] = (struct adpair){.dx = 1.0, .id = b.id};
        e->size = 2;
    }

    return ret;
}

/**
 * Adds variable a to double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_plus_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value + b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
        e->size = 1;
    }

    return ret;
}

/**
 * Adds double a to variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_plus_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a + b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Plus assign variable a and variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 */
inline void ad_plus_eq(__global struct gradient_structure* gs, struct variable* a, struct variable b) {
    a->value += b.value;

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = a->id};
        e->coeff[1] = (struct adpair){.dx = 1.0, .id = b.id};
        e->size = 2;
        e->id = a->id;
    }
}

inline void ad_plus_eq_g(__global struct gradient_structure* gs, __global struct variable* a, struct variable b) {
    a->value += b.value;

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = a->id};
        e->coeff[1] = (struct adpair){.dx = 1.0, .id = b.id};
        e->size = 2;
        e->id = a->id;
    }
}

/**
 * Plus assign variable a and double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 *  
 * @param gs
 * @param a
 * @param b
 */
inline void ad_plus_eq_d(__global struct gradient_structure* gs, struct variable* a, double b) {
    a->value += b;

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = a->id};
        e->size = 1;
        e->id = a->id;
    }
}

/**
 * Subtracts variable b from variable a. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_minus(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value - b.value, .id = 0};

    if (gs->recording) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
        e->coeff[1] = (struct adpair){.dx = -1.0, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Subtracts double b from variable a.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_minus_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value - b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
        e->size = 1;
    }

    return ret;
}

/**
 * Subtracts variable b from double a. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 *  
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_minus_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a - b.value, .id = 0};

    if (gs->recording) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = -1.0, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Multiplies to variables together. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_times(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value * b.value, .id = 0};


    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->id = ret.id;
        e->coeff[0] = (struct adpair){.dx = a.value, .id = a.id};
        e->coeff[1] = (struct adpair){.dx = b.value, .id = b.id};
        e->size = 2;
    }

    return ret;
}

/**
 * Multiplies variable a and double b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_times_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value * b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = a.value, .id = a.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Multiplies double a and variable b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_times_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a * b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = b.value, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Divides variable a by variable b.If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_divide(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value / b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b.value;
        e->coeff[0] = (struct adpair){.dx = inv, .id = a.id};
        e->coeff[1] = (struct adpair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

/**
 * Divides variable a by double b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed. 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_divide_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value / b, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b;
        e->coeff[0] = (struct adpair){.dx = inv, .id = a.id};
        e->size = 1;
        e->id = ret.id;
    }
    return ret;
}

/**
 * Divides double a by variable b. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable ad_divide_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a / b.value, .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / b.value;
        e->coeff[0] = (struct adpair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_cos(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = log(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = -1.0 * sin(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_sin(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = sin(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = cos(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_tan(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = tan(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = 1.0 / cos(v.value);
        e->coeff[0] = (struct adpair){.dx = temp*temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_acos(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = acos(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (-1.0) /
                pow(((1.0) -
                pow(v.value, (2.0))),
                (0.5));
        e->coeff[0] = (struct adpair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_asin(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = asin(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (1.0) /
                pow(((1.0) -
                pow(v.value, (2.0))),
                (0.5));
        e->coeff[0] = (struct adpair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_atan(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = atan(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (1.0) / (v.value * v.value + (1.0));
        e->coeff[0] = (struct adpair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_cosh(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = cosh(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = sinh(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_sinh(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = sinh(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = cosh(v.value), .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_tanh(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = tanh(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double temp = (1.0 / cosh(v.value))*(1.0 / cosh(v.value));
        e->coeff[0] = (struct adpair){.dx = temp, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_exp(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = exp(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        e->coeff[0] = (struct adpair){.dx = ret.value, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_log(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = log(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / v.value;
        e->coeff[0] = (struct adpair){.dx = inv, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_log10(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = log10(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = 1.0 / (v.value * 2.30258509299404590109361379290930926799774169921875);
        e->coeff[0] = (struct adpair){.dx = inv, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_pow(__global struct gradient_structure* gs,
        struct variable a, struct variable b) {
    struct variable ret = {.value = pow(a.value, b.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = b.value * pow(a.value, b.value - (1.0));
        e->coeff[0] = (struct adpair){.dx = inv, .id = a.id};
        e->coeff[1] = (struct adpair){.dx = log(a.value) * ret.value, .id = b.id};
        e->size = 2;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_pow_vd(__global struct gradient_structure* gs,
        struct variable a, double b) {
    struct variable ret = {.value = pow(a.value, b), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = b * pow(a.value, b - (1.0));
        e->coeff[0] = (struct adpair){.dx = inv, .id = a.id};
        e->size = 1;
        e->id = ret.id;
    }
    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_pow_dv(__global struct gradient_structure* gs,
        double a, struct variable b) {
    struct variable ret = {.value = pow(a, b.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = b.value * pow(a, b.value - (1.0));
        e->coeff[0] = (struct adpair){.dx = log(a) * ret.value, .id = b.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

inline const struct variable __attribute__((overloadable)) ad_sqrt(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = sqrt(v.value), .id = 0};

    if (gs->recording == 1) {
        int index = atomic_inc(&gs->counter);
        ret.id = index + gs->current_variable_id;
        __global struct entry* e =
                &gs->gradient_stack[index + gs->stack_current];
        double inv = .5 / ret.value;
        e->coeff[0] = (struct adpair){.dx = inv, .id = v.id};
        e->size = 1;
        e->id = ret.id;
    }

    return ret;
}

//
