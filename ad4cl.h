/* 
 * File:   adcl.h
 * Author: Matthew
 *
 * Created on January 22, 2015, 3:38 PM
 */

#ifndef ADCL_H
#define	ADCL_H
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef DEFAULT_ENTRY_SIZE
#define DEFAULT_ENTRY_SIZE 10000000
#endif

#ifndef MAX_VARIABLE_IN_EXPESSION
#define MAX_VARIABLE_IN_EXPESSION 2
#endif




//#define USE_ATOMICS

#ifdef USE_ATOMICS
#define atomic_inc(ptr) InterlockedIncrement(&ptr)-1
#else
#define atomic_inc(ptr) ptr++;
#endif


#ifdef	__cplusplus
extern "C" {
#endif

    struct /*__attribute__ ((packed))*/ ad_variable {
        double value;
        int id;
    };

    struct /*__attribute__ ((packed))*/ ad_pair {
        double dx;
        int id;
    };

    struct /*__attribute__ ((packed))*/ ad_entry {
        struct ad_pair coeff[MAX_VARIABLE_IN_EXPESSION];
        int id;
        int size;
    };

    struct /*__attribute__ ((packed))*/ ad_gradient_structure {
        struct ad_entry* gradient_stack;
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
    struct ad_gradient_structure* create_gradient_structure(int size) {
        struct ad_gradient_structure* gs = ( struct ad_gradient_structure*)malloc(sizeof (ad_gradient_structure));
        gs->current_variable_id = 0;
        gs->recording = 1;
        gs->stack_current = 0;
        gs->gradient_stack = (struct ad_entry*)malloc(sizeof (ad_entry) * size);
        return gs;
    }

    /**
     * To be called after the gradient_structure has been run in a 
     * OpenCL application. This does not need to be called if the 
     * gradient_structure has only been run on the host.
     * @param gs
     */
    inline void gpu_restore(struct ad_gradient_structure* gs) {
        gs->current_variable_id += gs->counter;
        gs->stack_current += gs->counter;
        gs->counter = 0;
    }

    
    inline void ad_init_var(struct ad_gradient_structure* gs, struct ad_variable* var, double value){
        var->id = atomic_inc(gs->current_variable_id);
        var->value = value;
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
    inline const struct ad_variable ad_plus(struct ad_gradient_structure* gs, struct ad_variable a, struct ad_variable b) {
        struct ad_variable ret = {.value = a.value + b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
            e.coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
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
    inline const struct ad_variable ad_plus_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
        struct ad_variable ret = {.value = a.value + b, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
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
    inline const struct ad_variable ad_plus_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
        struct ad_variable ret = {.value = a + b.value, .id = gs->current_variable_id++};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = b.id};
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
    inline void ad_plus_eq_v(struct ad_gradient_structure* gs, struct ad_variable* a, struct ad_variable b) {
        a->value += b.value;

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
            e.coeff[1] = (struct ad_pair){.dx = 1.0, .id = b.id};
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
    inline void ad_plus_eq_d(struct ad_gradient_structure* gs, struct ad_variable* a, double b) {
        a->value += b;

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a->id};
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
    inline const struct ad_variable ad_minus(struct ad_gradient_structure* gs, struct ad_variable a, struct ad_variable b) {
        struct ad_variable ret = {.value = a.value - b.value, .id = 0};

        if (gs->recording) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
            e.coeff[1] = (struct ad_pair){.dx = -1.0, .id = b.id};
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
    inline const struct ad_variable ad_minus_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
        struct ad_variable ret = {.value = a.value - b, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = 1.0, .id = a.id};
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
    inline const struct ad_variable ad_minus_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
        struct ad_variable ret = {.value = a - b.value, .id = 0};

        if (gs->recording) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = -1.0, .id = b.id};
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
    inline const struct ad_variable ad_times(struct ad_gradient_structure* gs, struct ad_variable a, struct ad_variable b) {
        struct ad_variable ret = {.value = a.value * b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = a.value, .id = a.id};
            e.coeff[1] = (struct ad_pair){.dx = b.value, .id = b.id};
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
    inline const struct ad_variable ad_times_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
        struct ad_variable ret = {.value = a.value * b, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = b, .id = a.id};
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
    inline const struct ad_variable ad_times_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
        struct ad_variable ret = {.value = a * b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = a, .id = b.id};
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
    inline const struct ad_variable ad_divide(struct ad_gradient_structure* gs, struct ad_variable a, struct ad_variable b) {
        struct ad_variable ret = {.value = a.value / b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = 1.0 / b.value;
            e.coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
            e.coeff[1] = (struct ad_pair){.dx = -1.0 * ret.value * inv, .id = b.id};
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
    inline const struct ad_variable ad_divide_vd(struct ad_gradient_structure* gs, struct ad_variable a, double b) {
        struct ad_variable ret = {.value = a.value / b, .id = gs->current_variable_id++};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = 1.0 / b;
            e.coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
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
    inline const struct ad_variable ad_divide_dv(struct ad_gradient_structure* gs, double a, struct ad_variable b) {
        struct ad_variable ret = {.value = a / b.value, .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = 1.0 / b.value;
            e.coeff[0] = (struct ad_pair){.dx = -1.0 * ret.value * inv, .id = b.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;

        }
        return ret;
    }

    inline const struct ad_variable ad_cos(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = log(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            //            double inv = 1.0 / v.value;
            e.coeff[0] = (struct ad_pair){.dx = -1.0 * sin(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_sin(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = sin(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            //            double inv = 1.0 / v.value;
            e.coeff[0] = (struct ad_pair){.dx = cos(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_tan(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = tan(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double temp = 1.0 / cos(v.value);
            e.coeff[0] = (struct ad_pair){.dx = temp*temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_acos(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = acos(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double temp = (-1.0) /
                    pow(((1.0) -
                    pow(v.value, (2.0))),
                    (0.5));
            e.coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_asin(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = asin(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double temp = (1.0) /
                    pow(((1.0) -
                    pow(v.value, (2.0))),
                    (0.5));
            e.coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_atan(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = atan(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double temp = (1.0) / (v.value * v.value + (1.0));
            e.coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_cosh(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = cosh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = sinh(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_sinh(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = sinh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = cosh(v.value), .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_tanh(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = tanh(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double temp = (1.0 / cosh(v.value))*(1.0 / cosh(v.value));
            e.coeff[0] = (struct ad_pair){.dx = temp, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_exp(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = exp(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            e.coeff[0] = (struct ad_pair){.dx = ret.value, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_log(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = log(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = 1.0 / v.value;
            e.coeff[0] = (struct ad_pair){.dx = 1.0 / v.value, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_log10(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = log10(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = 1.0 / (v.value * 2.30258509299404590109361379290930926799774169921875);
            e.coeff[0] = (struct ad_pair){.dx = inv, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_pow(struct ad_gradient_structure* gs,
            struct ad_variable a, struct ad_variable b) {
        struct ad_variable ret = {.value = pow(a.value, b.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = b.value * pow(a.value, b.value - (1.0));
            e.coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
            e.coeff[1] = (struct ad_pair){.dx = log(a.value) * ret.value, .id = b.id};
            e.size = 2;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_pow_vd(struct ad_gradient_structure* gs,
            struct ad_variable a, double b) {
        struct ad_variable ret = {.value = pow(a.value, b), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            ret.id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = b * pow(a.value, b - (1.0));
            e.coeff[0] = (struct ad_pair){.dx = inv, .id = a.id};
            e.size = 1;
            e.id = ret.id;
            //            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_pow_dv(struct ad_gradient_structure* gs,
            double a, struct ad_variable b) {
        struct ad_variable ret = {.value = pow(a, b.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = b.value * pow(a, b.value - (1.0));
            e.coeff[0] = (struct ad_pair){.dx = log(a) * ret.value, .id = b.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    inline const struct ad_variable ad_sqrt(struct ad_gradient_structure* gs, struct ad_variable v) {
        struct ad_variable ret = {.value = sqrt(v.value), .id = 0};

        if (gs->recording == 1) {
            int current = atomic_inc(gs->stack_current);
            int var_id = atomic_inc(gs->current_variable_id);
            /*__private*/ struct ad_entry e;
            double inv = .5 / ret.value;
            e.coeff[0] = (struct ad_pair){.dx = inv, .id = v.id};
            e.size = 1;
            e.id = var_id;
            ret.id = var_id;
            //barrier(CLK_LOCAL_MEM_FENCE);
            gs->gradient_stack[current] = e;
        }
        return ret;
    }

    struct ad_entry* create_entries(int size) {
        struct ad_entry* e = (struct ad_entry*) malloc(size * sizeof (ad_entry));

        for (int i = 0; i < size; i++) {
            e[i].id = 0;
            e[i].size = 0;
        }

        return e;

    }

    double* compute_gradient(struct ad_gradient_structure& gs, int& size) {


        double* gradient = NULL;
        if (gs.recording == 1) {
            gradient = (double*)malloc(sizeof (double)*(gs.current_variable_id + 1));
            size = gs.current_variable_id + 1;
            memset(gradient, 0, size * sizeof (double));
            //            for (int i = 0; i < size; i++) {
            //                gradient[i] = 0;
            //            }
            gradient[gs.gradient_stack[gs.stack_current - 1].id] = 1.0;

            for (int j = gs.stack_current - 1; j >= 0; j--) {
                int id = gs.gradient_stack[j].id;
                double w = gradient[id];
//                if (w != 0.0) {
                    gradient[id] = 0.0;
                    for (int i = 0; i < gs.gradient_stack[j].size; i++) {
                        //std::cout<<gs.gradient_stack[j].coeff[i].id<<"+="<<w<<"*"<<gs.gradient_stack[j].coeff[i].dx<<std::endl;
                        gradient[gs.gradient_stack[j].coeff[i].id] += w * gs.gradient_stack[j].coeff[i].dx;
                    }
//                }
            }
        }
        //        exit(0);
        return gradient;
    }



#ifdef	__cplusplus
}
#endif

#endif	/* ADCL_H */

