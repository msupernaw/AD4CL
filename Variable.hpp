/* 
 * File:   variable.hpp
 * Author: Matthew
 *
 * Created on January 27, 2015, 3:57 PM
 */

#ifndef VARIABLE_HPP
#define	VARIABLE_HPP
#include "ad4cl.h"
namespace ad4cl {

    /**
     * A c++ variable class to provide inter-operability between the native c
     * and OpenCL API's. Implements operator overloading. Template parameter
     * group_id creates variables with different gradient_structure's.
     */
    template<int group_id = 0 >
    class Variable {
    public:
        struct variable var;
        static struct gradient_structure* gs;

        Variable(const struct variable& v) {
            var = v;
        }

        operator double() {
            return var.value;
        }

        operator double() const {
            return var.value;
        }


    };

    template<int group_id>
    struct gradient_structure* Variable<group_id>::gs = create_gradient_structure(DEFAULT_ENTRY_SIZE);

    template<int group_id>
    inline const Variable<group_id> operator +(const Variable<group_id>& a, const Variable<group_id>& b) {
        return Variable<group_id>(plus_vv(a.gs, a.var, b.var));
    }

    template<int group_id>
    inline const Variable<group_id> operator +(const Variable<group_id>& a, const double& b) {
        return Variable<group_id>(plus_vd(a.gs, a.var, b));
    }

    template<int group_id>
    inline const Variable<group_id> operator +(const double& a, const Variable<group_id>& b) {
        return Variable<group_id>(plus_dv(b.gs, a, b.var));
    }

    template<int group_id>
    inline const Variable<group_id> operator -(const Variable<group_id>& a, const Variable<group_id>& b) {
        return Variable<group_id>(minus_vv(a.gs, a.var, b.var));
    }

    template<int group_id>
    inline const Variable<group_id> operator -(const Variable<group_id>& a, const double& b) {
        return Variable<group_id>(minus_vd(a.gs, a.var, b));
    }

    template<int group_id>
    inline const Variable<group_id> operator -(const double& a, const Variable<group_id>& b) {
        return Variable<group_id>(minus_dv(b.gs, a, b.var));
    }

    template<int group_id>
    inline const Variable<group_id> operator *(const Variable<group_id>& a, const Variable<group_id>& b) {
        return Variable<group_id>(times_vv(a.gs, a.var, b.var));
    }

    template<int group_id>
    inline const Variable<group_id> operator *(const Variable<group_id>& a, const double& b) {
        return Variable<group_id>(times_vd(a.gs, a.var, b));
    }

    template<int group_id>
    inline const Variable<group_id> operator *(const double& a, const Variable<group_id>& b) {
        return Variable<group_id>(times_dv(b.gs, a, b.var));
    }

    template<int group_id>
    inline const Variable<group_id> operator /(const Variable<group_id>& a, const Variable<group_id>& b) {
        return Variable<group_id>(divide_vv(a.gs, a.var, b.var));
    }

    template<int group_id>
    inline const Variable<group_id> operator /(const Variable<group_id>& a, const double& b) {
        return Variable<group_id>(divide_vd(a.gs, a.var, b));
    }

    template<int group_id>
    inline const Variable<group_id> operator /(const double& a, const Variable<group_id>& b) {
        return Variable<group_id>(divide_dv(b.gs, a, b.var));
    }




}


#endif	/* VARIABLE_HPP */

