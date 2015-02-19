#if !defined(_SIMPLE_)
#define _SIMPLE_

#define __CL_ENABLE_EXCEPTIONS 
#include "../../cl.hpp"

#include "../../ad4cl.h"


#define STACK_SIZE 5000000


#include <iostream>
#include <vector>

inline std::ostream& operator<<(std::ostream& out, const cl::Platform& platform) {
    out << "CL_PLATFORM_PROFILE    = " << platform.getInfo<CL_PLATFORM_PROFILE > () << "\n";
    out << "CL_PLATFORM_VERSION    = " << platform.getInfo<CL_PLATFORM_VERSION > () << "\n";
    out << "CL_PLATFORM_NAME       = " << platform.getInfo<CL_PLATFORM_NAME > () << "\n";
    out << "CL_PLATFORM_VENDOR     = " << platform.getInfo<CL_PLATFORM_VENDOR > () << "\n";
    out << "CL_PLATFORM_EXTENSIONS = " << platform.getInfo<CL_PLATFORM_EXTENSIONS > () << "\n";
    return out;
}

inline std::ostream& operator<<(std::ostream& out, const cl::Device& device) {
    out << "CL_DEVICE_ADDRESS_BITS                  = " << device.getInfo<CL_DEVICE_ADDRESS_BITS > () << "\n";
    out << "CL_DEVICE_AVAILABLE                     = " << device.getInfo<CL_DEVICE_AVAILABLE > () << "\n";
    out << "CL_DEVICE_COMPILER_AVAILABLE            = " << device.getInfo<CL_DEVICE_COMPILER_AVAILABLE > () << "\n";
    out << "CL_DEVICE_ENDIAN_LITTLE                 = " << device.getInfo<CL_DEVICE_ENDIAN_LITTLE > () << "\n";
    out << "CL_DEVICE_ERROR_CORRECTION_SUPPORT      = " << device.getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT > () << "\n";
    out << "CL_DEVICE_EXECUTION_CAPABILITIES        = " << device.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES > () << "\n";
    out << "CL_DEVICE_EXTENSIONS                    = " << device.getInfo<CL_DEVICE_EXTENSIONS > () << "\n";
    out << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE         = " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE > () << "\n";
    out << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE         = " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE > () << "\n";
    out << "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE     = " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE > () << "\n";
    out << "CL_DEVICE_GLOBAL_MEM_SIZE               = " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE > () << "\n";
    out << "CL_DEVICE_IMAGE_SUPPORT                 = " << device.getInfo<CL_DEVICE_IMAGE_SUPPORT > () << "\n";
    out << "CL_DEVICE_IMAGE2D_MAX_HEIGHT            = " << device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT > () << "\n";
    out << "CL_DEVICE_IMAGE2D_MAX_WIDTH             = " << device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH > () << "\n";
    out << "CL_DEVICE_IMAGE3D_MAX_DEPTH             = " << device.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH > () << "\n";
    out << "CL_DEVICE_IMAGE3D_MAX_HEIGHT            = " << device.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT > () << "\n";
    out << "CL_DEVICE_IMAGE3D_MAX_WIDTH             = " << device.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH > () << "\n";
    out << "CL_DEVICE_LOCAL_MEM_SIZE                = " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE > () << "\n";
    out << "CL_DEVICE_LOCAL_MEM_TYPE                = " << device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE > () << "\n";
    out << "CL_DEVICE_MAX_CLOCK_FREQUENCY           = " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY > () << "\n";
    out << "CL_DEVICE_MAX_COMPUTE_UNITS             = " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS > () << "\n";
    out << "CL_DEVICE_MAX_CONSTANT_ARGS             = " << device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS > () << "\n";
    out << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE      = " << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE > () << "\n";
    out << "CL_DEVICE_MAX_MEM_ALLOC_SIZE            = " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE > () << "\n";
    out << "CL_DEVICE_MAX_PARAMETER_SIZE            = " << device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE > () << "\n";
    out << "CL_DEVICE_MAX_READ_IMAGE_ARGS           = " << device.getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS > () << "\n";
    out << "CL_DEVICE_MAX_SAMPLERS                  = " << device.getInfo<CL_DEVICE_MAX_SAMPLERS > () << "\n";
    out << "CL_DEVICE_MAX_WORK_GROUP_SIZE           = " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE > () << "\n";
    out << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS      = " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS > () << "\n";
    out << "CL_DEVICE_MAX_WRITE_IMAGE_ARGS          = " << device.getInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS > () << "\n";
    out << "CL_DEVICE_MEM_BASE_ADDR_ALIGN           = " << device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN > () << "\n";
    out << "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE      = " << device.getInfo<CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE > () << "\n";
    out << "CL_DEVICE_NAME                          = " << device.getInfo<CL_DEVICE_NAME > () << "\n";
    out << "CL_DEVICE_PLATFORM                      = " << device.getInfo<CL_DEVICE_PLATFORM > () << "\n";
    out << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR   = " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR > () << "\n";
    out << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE > () << "\n";
    out << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT  = " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT > () << "\n";
    out << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT    = " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT > () << "\n";
    out << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG   = " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG > () << "\n";
    out << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT  = " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT > () << "\n";
    out << "CL_DEVICE_PROFILE                       = " << device.getInfo<CL_DEVICE_PROFILE > () << "\n";
    out << "CL_DEVICE_PROFILING_TIMER_RESOLUTION    = " << device.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION > () << "\n";
    out << "CL_DEVICE_QUEUE_PROPERTIES              = " << device.getInfo<CL_DEVICE_QUEUE_PROPERTIES > () << "\n";
    out << "CL_DEVICE_SINGLE_FP_CONFIG              = " << device.getInfo<CL_DEVICE_SINGLE_FP_CONFIG > () << "\n";
    out << "CL_DEVICE_TYPE                          = " << device.getInfo<CL_DEVICE_TYPE > () << "\n";
    out << "CL_DEVICE_VENDOR_ID                     = " << device.getInfo<CL_DEVICE_VENDOR_ID > () << "\n";
    out << "CL_DEVICE_VENDOR                        = " << device.getInfo<CL_DEVICE_VENDOR > () << "\n";
    out << "CL_DEVICE_VERSION                       = " << device.getInfo<CL_DEVICE_VERSION > () << "\n";
    out << "CL_DRIVER_VERSION                       = " << device.getInfo<CL_DRIVER_VERSION > () << "\n";
    return out;
}

class model_data : public ad_comm {
    data_int nobs;
    data_int method;
    data_int ad4cl_stack_size;
    init_adstring  ad4cl_api;
    init_adstring kernel_code;
    data_int gpu_index;
    double A;
    double B;
    double S;
    double* Y;
    double* x;
    dvector YY;
    dvector XX;
    ~model_data();
    model_data(int argc, char * argv[]);
    friend class model_parameters;
};

class model_parameters : public model_data,
public function_minimizer {
public:
    ~model_parameters();
    void preliminary_calculations(void);
    void set_runtime(void);

    virtual void * mycast(void) {
        return (void*) this;
    }

    static int mc_phase(void) {
        return initial_params::mc_phase;
    }

    static int mceval_phase(void) {
        return initial_params::mceval_phase;
    }

    static int sd_phase(void) {
        return initial_params::sd_phase;
    }

    static int current_phase(void) {
        return initial_params::current_phase;
    }

    static int last_phase(void) {
        return (initial_params::current_phase
                >= initial_params::max_number_phases);
    }

    static prevariable current_feval(void) {
        return *objective_function_value::pobjfun;
    }
private:

    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::Context context;
    cl::Program program_;
    cl::Program::Sources source;
    std::vector<cl::Device> devices;
    cl_int error;
    cl::Buffer gs_d;
    cl::Buffer ad_entry_d;
    cl::Buffer a_d;
    cl::Buffer b_d;
    cl::Buffer x_d;
    cl::Buffer y_d;
    cl::Buffer out_d;

    int DATA_SIZE;

    struct ad_variable aa;
    struct ad_variable bb;
    struct ad_variable sum;
    struct ad_variable* out;
    struct ad_gradient_structure* gs;
    struct ad_entry* gradient_stack;
    size_t global_size, local_size;
    std::vector<double> gradient;

    enum GradientMethod {
        ADMB = 0,
        AD4CL_DEVICE,
        AD4CL_HOST
    };

    GradientMethod gradient_method;

    ivector integer_control_flags;
    dvector double_control_flags;
    param_init_number a;
    param_init_number b;
    param_vector pred_Y;
    param_number prior_function_value;
    param_number likelihood_function_value;
    objective_function_value f;
public:
    void initialize_opencl(void);
    virtual void userfunction(void);
    virtual void report(const dvector& gradients);
    virtual void final_calcs(void);
    model_parameters(int sz, int argc, char * argv[]);

    virtual void initializationfunction(void) {
    }

};
#endif
