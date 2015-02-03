/* 
 * File:   main.cpp
 * Author: Matthew
 *
 * Simple example of using the GPU for reverse mode
 * Automatic differentiation.
 * 
 * Created on January 21, 2015, 9:01 AM
 */

#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

#define __CL_ENABLE_EXCEPTIONS 

#include "cl.hpp"
#include "ad4cl.h"
#include "Variable.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

//reserved size of the stack
#define STACK_SIZE 10000000 

#define CL_PROFILING


using namespace std;

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


//simple kernel to compute the sum of ((a*x[i] + b)-y[i])^2
std::string my_kernel = "__kernel void AD(__global struct gradient_structure* gs,\n"\
       " __global  struct entry*  gradient_stack,\n"\
       "  const __global struct  variable* a,\n"\
       "  const __global struct  variable*b,\n"\
       " const __global  double  *x,\n"\
       " const __global  double  *y,\n"\
       " __global struct variable *out, int size) {\n"\
        "struct local_gradient_structure lgs;\n"\
   " int id = get_global_id(0);\n"\
        " struct variable al = *a;\n"\
        " struct variable bl = *b;\n"\
   " if(id==0){init(gs, gradient_stack);barrier(CLK_GLOBAL_MEM_FENCE);}\n"\
        " if (id < size) {\n"\
   "     double xx = x[id];\n"\
   "     double yy = y[id];\n"\
        "struct variable temp =minus_vd(gs, plus(gs, times_vd(gs, al, xx), bl), yy);\n"\
   "     struct variable v = pow_d(gs, temp,2.0);\n"\
   "     out[id] = v;\n"\
   " }\n/*if(get_local_id(0)){gs->counter+=lgs.current_variable_id++;}*/"\
  "}\n";

void AD(struct gradient_structure* gs,
        struct variable* a,
        struct variable*b,
        double *x,
        double *y,
        struct variable *out, int size) {

    //    int id = get_global_id(0);
    for (int i = 0; i < size; i++) {
        //minus(gs, plus(gs, times(gs,a, x[i]) ,b), y[i]);
        struct variable temp =  minus_vd(gs, plus_vv(gs, times_vd(gs, *a, x[i]), *b), y[i]);
        struct variable v = times_vv(gs,temp,temp);// minus_vd(gs, plus_vv(gs, times_vd(gs, *a, x[i]), *b), y[i]), minus_vd(gs, plus_vv(gs, times_vd(gs, *a, x[i]), *b), y[i]));
        out[i] = v;
        //        std::cout << out[i].value << " === " << std::pow(((a->value * x[i] + b->value) - y[i]), 2.0) << "\n";
    }


}
#include <sys/time.h>
int HOST = 0;

void TEST_EXPRESSION() {




}



#include <limits>
/**
 * Simple example of running a ad4cl kernel.
 * 
 * Demonstrates how recording can be turned on and off.
 * 
 * Shows how to compute a gradient vector and get the 
 * derivative w.r.t a variable.
 */
int main(int argc, char** argv) {
    std::cout << sizeof (struct gradient_structure) << "\n" << sizeof (struct entry);
    std::cout << "\n" << 49000 / 40 << "\n";

    std::string source_code;

    //Read the ad4cl api.
    std::string line;
    std::ifstream in;
    in.open("ad.cl");

    std::stringstream ss;

    while (in.good()) {
        std::getline(in, line);
        ss << line << "\n";
    }
    
     std::ifstream kin;
    kin.open("kernel.cl");

    while (kin.good()) {
        std::getline(kin, line);
        ss << line << "\n";
    }
    

    //append with our kernel
//    ss << my_kernel;
    source_code = ss.str();

    std::cout<<source_code<<"\n";



    //opencl declarations
    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::Context context;
    cl::Program program_;
    std::vector<cl::Device> devices;
    cl_int error = CL_SUCCESS;

    // Query platforms
    std::vector<cl::Platform> platforms;

    try {
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            exit(0);
        }

        //print platform info 
        std::cout << platforms[1];

        // Get list of devices on default platform and create context
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0])(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
        devices = context.getInfo<CL_CONTEXT_DEVICES > ();


        const cl::Device device = devices[0];

        //print device info
        std::cout << device << "\n";

        //set the program source
        cl::Program::Sources source(1, std::make_pair(source_code.c_str(), source_code.size()));
        program_ = cl::Program(context, source, &error);

        //build the program
        program_.build(devices);

        //set the queue
#ifdef CL_PROFILING
        queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
#else
        queue = cl::CommandQueue(context, devices[0]);
#endif
        if (error != CL_SUCCESS) {
            std::cout << "---> " << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG > (devices[0]) << "\n";
            exit(0);
        }


        // Create kernel object
        kernel = cl::Kernel(program_, "AD");

    } catch (cl::Error err) {
        std::cout << "---> " << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG > (devices[0]);
    }


    size_t global_size, local_size;

    //initialize the data set
    int DATA_SIZE = 700003;
    double* x = new double[DATA_SIZE];
    double* y = new double[DATA_SIZE];

    // Number of work items in each local work group
    local_size = 64;//devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE > ();

    // Number of total work items - localSize must be devisor
    global_size = std::ceil(DATA_SIZE / (float) local_size + 1) * local_size;
std::cout << global_size << "\n";
    //    exit(0);
    double aa = 4.1919;
    double bb = 3.2123;

    //set the data
    for (int i = 0; i < DATA_SIZE; i++) {
        x[i] = static_cast<double> (i);
        y[i] = aa * x[i] + bb;
    }

    //create a gradient structure
    struct gradient_structure gs;
    gs.current_variable_id = 0;
    gs.stack_current = 0;
    gs.recording = 1;
    gs.counter = 0;
    gs.max_entries_per_kernel = 9;
    struct entry* entries = new entry[STACK_SIZE];
    for (int i = 0; i < STACK_SIZE; i++) {
        entries[i].size = 0;
        entries[i].id = 0;
    }
    gs.gradient_stack = entries;

    //create out variables
    variable a = {.value = aa - .005, .id = gs.current_variable_id++};
    variable b = {.value = bb - .0051, .id = gs.current_variable_id++};
    variable sum = {.value = 0.0, .id = gs.current_variable_id++};
    variable* out = new variable[DATA_SIZE]; //{.value = 0.0, .id = gs.current_variable_id++};


    try {


        //set the buffers
        cl::Buffer gs_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof (gradient_structure));
        cl::Buffer entry_d = cl::Buffer(context, CL_MEM_READ_WRITE, STACK_SIZE * sizeof (entry), entries);
        cl::Buffer a_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof (variable));
        cl::Buffer b_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof (variable));
        cl::Buffer x_d = cl::Buffer(context, CL_MEM_READ_ONLY, DATA_SIZE * sizeof (double));
        cl::Buffer y_d = cl::Buffer(context, CL_MEM_READ_ONLY, DATA_SIZE * sizeof (double));
        cl::Buffer out_d = cl::Buffer(context, CL_MEM_WRITE_ONLY, DATA_SIZE * sizeof (variable));

        queue.enqueueWriteBuffer(x_d, CL_TRUE, 0, sizeof (double)*DATA_SIZE, x);
        queue.enqueueWriteBuffer(y_d, CL_TRUE, 0, sizeof (double)*DATA_SIZE, y);
        //        queue.enqueueWriteBuffer(out_d, CL_TRUE, 0, sizeof (variable), &out);

        kernel.setArg(0, gs_d);
        kernel.setArg(1, entry_d);
        kernel.setArg(2, a_d);
        kernel.setArg(3, b_d);
        kernel.setArg(4, x_d);
        kernel.setArg(5, y_d);
        kernel.setArg(6, out_d);
        kernel.setArg(7, DATA_SIZE);
        //        kernel.setArg(8, DATA_STRIDE);
        // Number of work items in each local work group
        cl::NDRange localSize(local_size);
        // Number of total work items - localSize must be devisor
        cl::NDRange globalSize(global_size); //(int) (std::ceil(DATA_SIZE / (double) 64)*64));



        for (int iter = 0; iter < 36; iter++) {
            sum.value = 0.0;
            std::cout << "iteration " << iter << std::endl;
            if ((iter % 2) == 0) {
                gs.recording = 1;
            } else {
                gs.recording = 1;
            }

            if (HOST) {
                static struct timeval tm1, tm2;
                gettimeofday(&tm1, NULL);
                AD(&gs, &a, &b, x, y, out, DATA_SIZE);
                gettimeofday(&tm2, NULL);
                unsigned long long t = 1000 * (tm2.tv_sec - tm1.tv_sec) + (tm2.tv_usec - tm1.tv_usec) / 1000;
                printf("%llu ms\n", t);
            } else {

                queue.enqueueWriteBuffer(gs_d, CL_TRUE, 0, sizeof (gradient_structure), &gs);
                queue.enqueueWriteBuffer(a_d, CL_TRUE, 0, sizeof (variable), &a);
                queue.enqueueWriteBuffer(b_d, CL_TRUE, 0, sizeof (variable), &b);
                //            out.value = 0.0;
                queue.enqueueWriteBuffer(out_d, CL_TRUE, 0, DATA_SIZE * sizeof (variable), out);
                cl::Event event;
                queue.enqueueNDRangeKernel(
                        kernel,
                        cl::NullRange,
                        globalSize,
                        localSize,
                        NULL,
                        &event);





                // Block until kernel completion
                event.wait();

#ifdef CL_PROFILING

                cl_ulong start =
                        event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                cl_ulong end =
                        event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                double time = 1.e-6 * (end - start);
                double startTime = start * 1.e-6;
                double endTime = end * 1.e-6;
                cout << "Kernel (start,end) " << startTime << "," << endTime
                        << " Time for kernel to execute " << time << std::endl;
#endif
            }


            //our function value.
            struct variable f;
            if (!HOST) {
                //read our kernel value
                queue.enqueueReadBuffer(out_d, CL_TRUE, 0, DATA_SIZE * sizeof (variable), (struct variable*) out);

            }

            if (gs.recording == 1) {
                double* g;
                int gsize = 0;

                if (!HOST) {
                    queue.enqueueReadBuffer(gs_d, CL_TRUE, 0, sizeof (gradient_structure), &gs);
                    queue.enqueueReadBuffer(entry_d, CL_TRUE, 0, STACK_SIZE * sizeof (entry), (struct entry*) entries);
                    gs.gradient_stack = entries;

                    gs.stack_current += gs.counter;
                    gs.current_variable_id += gs.counter;
                    std::cout << gs.current_variable_id << "\n";
                    std::cout << gs.stack_current << std::endl;

                }


                for (int i = 0; i < DATA_SIZE; i++) {
                    plus_eq_v(&gs, &sum, out[i]/*times_vv(&gs,out[i],out[i])*/);
                    //                    sum = plus_vv(&gs, sum, out[i]);
                    //                std::cout<<sum.value<<"\n";
                    out[i].value = 0;
                }
                //finish up with the native api.
                f = times_dv(&gs, static_cast<double> (DATA_SIZE) / 2.0, ad_log(&gs, sum));

                //                break;
                //compute the function gradient
                g = compute_gradient(gs, gsize);



                //print function value a derivatives w.r.t a and b.
                std::cout << std::fixed << std::setprecision(10) << "f  = " << f.value << std::endl;
                std::cout << a.value << ", df/da = " << g[a.id] << std::endl;
                std::cout << b.value << ", df/db = " << g[b.id] << std::endl;
                free(g);
            } else {

                for (int i = 0; i < DATA_SIZE; i++) {
                    plus_eq_v(&gs, &sum, out[i]);
                    //                    sum = plus_vv(&gs, sum, out[i]);
                    //                std::cout<<sum.value<<"\n";
                    out[i].value = 0;
                }
                //finish up with the native api.
                f = times_dv(&gs, static_cast<double> (DATA_SIZE) / 2.0, ad_log(&gs, sum));

                // print function value a derivatives w.r.t a and b.
                std::cout << " f  = " << f.value << std::endl;
                std::cout << a.value << ", df/da = " << 0 << std::endl;
                std::cout << b.value << ", df/db = " << 0 << std::endl;
            }

            //tweak the values of the independent variables
            a.value += .0000001;
            b.value += .0000001;

            gs.stack_current = 0;
            gs.current_variable_id = b.id + 1;
            gs.counter = 1;
            for (int i = 0; i < STACK_SIZE; i++) {
                entries[i].size = 0;
                entries[i].id = 0;
            }
        }

    } catch (cl::Error err) {

        std::cout << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG > (devices[0]);
        exit(0);
    }

    delete[] x;
    delete[] y;
    delete[] entries;

    return 0;
}

