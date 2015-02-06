/* 
 * File:   adcl.h
 * Author: Matthew
 *
 * Created on January 22, 2015, 3:38 PM
 */

#ifndef ADCL_H
#define	ADCL_H
#include <math.h>
#include <malloc.h>
#include <stdint.h>

#ifndef DEFAULT_ENTRY_SIZE
#define DEFAULT_ENTRY_SIZE 10000000
#endif

#ifndef MAX_VARIABLE_IN_EXPESSION
#define MAX_VARIABLE_IN_EXPESSION 2
#endif




#define USE_ATOMICS

#ifdef USE_ATOMICS
#define atomic_inc(ptr) InterlockedIncrement(&ptr)-1
#else
#define atomic_inc(ptr) ptr++;
#endif


#ifdef	__cplusplus
extern "C" {
#endif

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
        struct entry* gradient_stack;
        int current_variable_id;
        int stack_current;
        int recording;
        int counter;
    };

    /**
     * Creates a new gradient_structure.
     * @param size - length of the entries array.
     * @return 
     */
    struct gradient_structure* create_gradient_structure(int size) {
        struct gradient_structure* gs = malloc(sizeof (gradient_structure));
        gs->current_variable_id = 0;
        gs->recording = 1;
        gs->stack_current = 0;
        gs->gradient_stack = malloc(sizeof (entry) * size);
    }

    /**
     * To be called after the gradient_structure has been run in a 
     * OpenCL application. This does not need to be called if the 
     * gradient_structure has only been run on the host.
     * @param gs
     */
    inline void gpu_restore(struct gradient_structure* gs) {
        gs->current_variable_id += gs->counter;
        gs->stack_current += gs->counter;
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
    inline const struct variable ad_plus(struct gradient_structure* gs, struct variable a, struct variable b) {
        struct variable ret = {.value = a.value + b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_plus_vd(struct gradient_structure* gs, struct variable a, double b) {
        struct variable ret = {.value = a.value + b, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct entry e;
            e.coeff[0] = (struct adpair){.dx = 1.0, .id = a.id};
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
    inline const struct variable ad_plus_dv(struct gradient_structure* gs, double a, struct variable b) {
        struct variable ret = {.value = a + b.value, .id = gs->current_variable_id++};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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
    inline void ad_plus_eq_v(struct gradient_structure* gs, struct variable* a, struct variable b) {
        a->value += b.value;

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
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
    inline void ad_plus_eq_d(struct gradient_structure* gs, struct variable* a, double b) {
        a->value += b;

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
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
    inline const struct variable ad_minus(struct gradient_structure* gs, struct variable a, struct variable b) {
        struct variable ret = {.value = a.value - b.value, .id = 0};

        if (gs->recording) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_minus_vd(struct gradient_structure* gs, struct variable a, double b) {
        struct variable ret = {.value = a.value - b, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_minus_dv(struct gradient_structure* gs, double a, struct variable b) {
        struct variable ret = {.value = a - b.value, .id = 0};

        if (gs->recording) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_times(struct gradient_structure* gs, struct variable a, struct variable b) {
        struct variable ret = {.value = a.value * b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct entry e;
            e.coeff[0] = (struct adpair){.dx = a.value, .id = a.id};
            e.coeff[1] = (struct adpair){.dx = b.value, .id = b.id};
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
    inline const struct variable ad_times_vd(struct gradient_structure* gs, struct variable a, double b) {
        struct variable ret = {.value = a.value * b, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_times_dv(struct gradient_structure* gs, double a, struct variable b) {
        struct variable ret = {.value = a * b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_divide(struct gradient_structure* gs, struct variable a, struct variable b) {
        struct variable ret = {.value = a.value / b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_divide_vd(struct gradient_structure* gs, struct variable a, double b) {
        struct variable ret = {.value = a.value / b, .id = gs->current_variable_id++};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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
    inline const struct variable ad_divide_dv(struct gradient_structure* gs, double a, struct variable b) {
        struct variable ret = {.value = a / b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_cos(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = log(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_sin(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = sin(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_tan(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = tan(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_acos(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = acos(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_asin(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = asin(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_atan(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = atan(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_cosh(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = cosh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_sinh(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = sinh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_tanh(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = tanh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_exp(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = exp(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_log(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = log(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_log10(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = log10(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
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

    inline const struct variable __attribute__((overloadable)) ad_pow(struct gradient_structure* gs,
            struct variable a, struct variable b) {
        struct variable ret = {.value = pow(a.value, b.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct entry e;
            double inv = b.value * pow(a.value, b.value - (1.0));
            e.coeff[0] = (struct adpair){.dx = inv, .id = a.id};
            e.coeff[1] = (struct adpair){.dx = log(a.value) * ret.value, .id = b.id};
            e.size = 2;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) ad_pow_vd(struct gradient_structure* gs,
            struct variable a, double b) {
        struct variable ret = {.value = pow(a.value, b), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct entry e;
            double inv = b * pow(a.value, b - (1.0));
            e.coeff[0] = (struct adpair){.dx = inv, .id = a.id};
            e.size = 1;
            e.id = ret.id;
            //            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) ad_pow_dv(struct gradient_structure* gs,
            double a, struct variable b) {
        struct variable ret = {.value = pow(a, b.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct entry e;
            double inv = b.value * pow(a, b.value - (1.0));
            e.coeff[0] = (struct adpair){.dx = log(a) * ret.value, .id = b.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct variable __attribute__((overloadable)) ad_sqrt(struct gradient_structure* gs, struct variable v) {
        struct variable ret = {.value = sqrt(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct entry e;
            double inv = .5 / ret.value;
            e.coeff[0] = (struct adpair){.dx = inv, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    struct entry* create_entries(int size) {
        struct entry* e = (struct entry*) malloc(size * sizeof (entry));

        for (int i = 0; i < size; i++) {
            e[i].id = 0;
            e[i].size = 0;
        }

        return e;

    }

    double* compute_gradient(struct gradient_structure& gs, int& size) {


        double* gradient = NULL;
        if (gs.recording == 1) {
            gradient = malloc(sizeof (double)*(gs.current_variable_id + 1));
            size = gs.current_variable_id + 1;
            memset(gradient, 0, size * sizeof (double));
            //            for (int i = 0; i < size; i++) {
            //                gradient[i] = 0;
            //            }

            gradient[gs.gradient_stack[gs.stack_current - 1].id] = 1.0;

            for (int j = gs.stack_current - 1; j >= 0; j--) {
                int id = gs.gradient_stack[j].id;
                double w = gradient[id];
                if (w != 0.0) {
                    gradient[id] = 0.0;
                    for (int i = 0; i < gs.gradient_stack[j].size; i++) {
                        //                    std::cout<<gs.gradient_stack[j].coeff[i].id<<"+="<<w<<"*"<<gs.gradient_stack[j].coeff[i].dx<<std::endl;
                        gradient[gs.gradient_stack[j].coeff[i].id] += w * gs.gradient_stack[j].coeff[i].dx;
                    }
                }
            }
        }
        return gradient;
    }

  

#ifdef	__cplusplus
}
#endif

#endif	/* ADCL_H */

