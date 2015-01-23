/**
 * A simple API to achieve reverse mode automatic differentiation of computer 
 * programs on a GPU using OpenCL. 
 */


#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision doubleing point not supported by OpenCL implementation."
#endif



struct variable {
    double value;
    int id;
};

struct adpair {
    double dx;
    int id;
};

struct entry {
    struct adpair coeff[2];
    int id;
    int size;
};

struct gradient_structure {
    __global struct entry* gradient_stack;
    int current_variable_id;
    int stack_current;
    int recording;
};

/**
 * Adds two variables together. If the gradient structure is recording, 
 * entries will be added, otherwise the result is only computed.
 * 
 * @param gs
 * @param a
 * @param b
 * @return 
 */
inline const struct variable plus(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value + b.value, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        ret.id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
        e.coeff[1] = (struct adpair){.dx = 1.0, .id = b.id};
        e.size = 2;
        e.id = ret.id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable plus_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value + b, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0,.id = a.id};
        e.size = 1;
        e.id = ret.id;
        ret.id = var_id;
        e.size = 2;
        e.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable plus_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a + b.value, .id = atomic_inc(&gs->current_variable_id)};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0, .id = b.id};
        e.size = 1;
        ret.id = var_id;
        e.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline void plus_eq(__global struct gradient_structure* gs, struct variable* a, struct variable b) {
    a->value += b.value;

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0, .id = a->id};
        e.coeff[1] = (struct adpair){.dx = 1.0, .id = b.id};
        e.size = 2;
        e.id = a->id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
    }
}

inline void plus_eq_g(__global struct gradient_structure* gs, __global struct variable* a, struct variable b) {
    barrier(CLK_LOCAL_MEM_FENCE);
    a->value += b.value;

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0, .id = a->id};
        e.coeff[1] = (struct adpair){.dx = 1.0, .id = b.id};
        e.size = 2;
        e.id = a->id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline void plus_eq_d(__global struct gradient_structure* gs, struct variable* a, double b) {
    a->value += b;

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0, .id = a->id};
        e.size = 1;
        e.id = a->id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable minus(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value - b.value, .id = 0};

    if (gs->recording) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
        e.coeff[1] = (struct adpair){.dx = -1.0, .id = b.id};
        e.size = 2;
        ret.id = var_id;
        e.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable minus_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value - b, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        ret.id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
        e.size = 1;
        e.id = ret.id;

        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable minus_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a - b.value, .id = 0};

    if (gs->recording) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = -1.0, .id = b.id};
        e.size = 1;
        e.id = var_id;
        ret.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable times(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value * b.value, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        ret.id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = a.value,.id = a.id};
        e.coeff[1] = (struct adpair){.dx = b.value,.id = b.id};
        e.size = 2;
        e.id = ret.id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable times_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value * b, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = a.value, .id = a.id};
        e.size = 1;
        ret.id = var_id;
        e.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable times_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a * b.value, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        e.coeff[0] = (struct adpair){.dx = b.value, .id = b.id};
        e.size = 1;
        e.id = var_id;
        ret.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable divide(__global struct gradient_structure* gs, struct variable a, struct variable b) {
    struct variable ret = {.value = a.value / b.value, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        double inv = 1.0 / b.value;
        e.coeff[0] = (struct adpair){.dx = inv, .id = a.id};
        e.coeff[1] = (struct adpair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e.size = 2;
        e.id = var_id;
        ret.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable divide_vd(__global struct gradient_structure* gs, struct variable a, double b) {
    struct variable ret = {.value = a.value / b, .id = atomic_inc(&gs->current_variable_id)};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        double inv = 1.0 / b;
        e.coeff[0] = (struct adpair){.dx = inv, .id = a.id};
        e.size = 1;
        e.id = var_id;
        ret.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
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
inline const struct variable divide_dv(__global struct gradient_structure* gs, double a, struct variable b) {
    struct variable ret = {.value = a / b.value, .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        double inv = 1.0 / b.value;
        e.coeff[0] = (struct adpair){.dx = -1.0 * ret.value * inv, .id = b.id};
        e.size = 1;
        e.id = var_id;
        ret.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;

    }
    return ret;
}

inline const struct variable __attribute__((overloadable)) log(__global struct gradient_structure* gs, struct variable v) {
    struct variable ret = {.value = log(v.value), .id = 0};

    if (gs->recording == 1) {
        int current = atomic_inc(&gs->stack_current);
        int var_id = atomic_inc(&gs->current_variable_id);
        /*__private*/ struct entry e;
        double inv = 1.0 / v.value;
        e.coeff[0] = (struct adpair){.dx = inv,.id = v.id};
        e.size = 1;
        e.id = var_id;
        ret.id = var_id;
        //barrier(CLK_LOCAL_MEM_FENCE);
        gs->gradient_stack[current] = e;
    }
    return ret;
}

 inline const struct variable __attribute__((overloadable)) cos(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = log(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            //            double inv = 1.0 / v.value;
            e.coeff[0] = (struct adpair){.dx = -1.0 * sin(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) sin(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = sin(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            //            double inv = 1.0 / v.value;
            e.coeff[0] = (struct adpair){.dx = cos(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) tan(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = tan(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double temp = 1.0 / cos(v.value);
            e.coeff[0] = (struct adpair){.dx = temp*temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) acos(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = acos(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double temp = (-1.0) /
                    pow(((1.0) -
                    pow(v.value, (2.0))),
                    (0.5));
            e.coeff[0] = (struct adpair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) asin(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = asin(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double temp = (1.0) /
                    pow(((1.0) -
                    pow(v.value, (2.0))),
                    (0.5));
            e.coeff[0] = (struct adpair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) atan(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = atan(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double temp = (1.0) / (v.value * v.value + (1.0));
            e.coeff[0] = (struct adpair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) cosh(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = cosh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            e.coeff[0] = (struct adpair){.dx = sinh(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) sinh(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = sinh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            e.coeff[0] = (struct adpair){.dx = cosh(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) tanh(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = tanh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double temp = (1.0 / cosh(v.value))*(1.0 / cosh(v.value));
            e.coeff[0] = (struct adpair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) exp(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = exp(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            e.coeff[0] = (struct adpair){.dx = ret.value, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) log(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = log(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double inv = 1.0 / v.value;
            e.coeff[0] = (struct adpair){.dx = inv, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) log10(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = log10(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double inv = 1.0 / (v.value * 2.30258509299404590109361379290930926799774169921875);
            e.coeff[0] = (struct adpair){.dx = inv, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) pow(struct gradient_structure* gs,
            struct variable a, struct variable b) {
        struct variable ret = {.value = pow(a.value,b.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double inv = b.value * pow(a.value, b.value - (1.0));
            e.coeff[0] = (struct adpair){.dx = inv, .id = a.id};
            e.coeff[1] = (struct adpair){.dx = log(a.value)*ret.value, .id = b.id};
            e.size = 2;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }
    
    inline const struct variable __attribute__((overloadable)) pow_d(struct gradient_structure* gs,
            struct variable a, double b) {
        struct variable ret = {.value = pow(a.value,b), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double inv = b * pow(a.value, b - (1.0));
            e.coeff[0] = (struct adpair){.dx = inv, .id = a.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }
    
    inline const struct variable __attribute__((overloadable)) d_pow(struct gradient_structure* gs,
            double a, struct variable b) {
        struct variable ret = {.value = pow(a,b.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double inv = b.value * pow(a, b.value - (1.0));
            e.coeff[0] = (struct adpair){.dx = log(a)*ret.value, .id = b.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }
    
    inline const struct variable __attribute__((overloadable)) sqrt(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = sqrt(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = gs->stack_current++;
            int var_id = gs->current_variable_id++;
            /*__private*/ struct entry e;
            double inv = .5/ ret.value;
            e.coeff[0] = (struct adpair){.dx = inv, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

