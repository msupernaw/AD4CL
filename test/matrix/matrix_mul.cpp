/* 
 * File:   matrix_mul.cpp
 * Author: Matthew
 *
 * Created on February 19, 2015, 2:13 PM
 */
#define __CL_ENABLE_EXCEPTIONS 

#define CL_PROFILING

//#define HOST

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>

#include "../../ad4cl.h"
#include "../../cl.hpp"


#define widthA 128
#define heightA 128

#define widthB heightA
#define heightB 128

#define widthC widthA
#define heightC heightB

#define GRADIENT_STACK_SIZE 10000000

using namespace std;

void MatrixMultHost(struct ad_gradient_structure* gs,
        struct ad_variable* A,
        struct ad_variable* B,
        struct ad_variable* C) {


    for (int i = 0; i < widthA; i++) {
        for (int j = 0; j < heightC; j++) {
            struct ad_variable value;
            ad_init_var(gs, &value, 0.0);
            for (int k = 0; k < widthB; k++) {
                ad_plus_eq_v(gs, &value, ad_times(gs, A[k + j * widthA], B[k * widthB + i]));
            }
            C[i * heightC + j] = value;
        }

    }
}

/*
 * 
 */
int main(int argc, char** argv) {

    struct ad_gradient_structure* gs; // = create_gradient_structure(GRADIENT_STACK_SIZE);
    int lastid;
    gs = new ad_gradient_structure();
    gs->current_variable_id = 1;
    gs->stack_current = 0;
    gs->recording = 1;
    gs->counter = 1;

    struct ad_entry* gradient_stack = new ad_entry[GRADIENT_STACK_SIZE];
    for (int i = 0; i < GRADIENT_STACK_SIZE; i++) {
        gradient_stack[i].size = 0;
        gradient_stack[i].id = 0;
    }
    gs->gradient_stack = gradient_stack;


    struct ad_variable * A = new struct ad_variable[widthA * heightA];
    struct ad_variable * B = new struct ad_variable[widthB * heightB];
    struct ad_variable * C = new struct ad_variable[widthC * heightC];

    for (int i = 0; i < widthA * heightA; i++) {
        ad_init_var(gs, &A[i], ((double) rand() / (RAND_MAX + 1)));
    }

    for (int i = 0; i < widthB * heightB; i++) {
        ad_init_var(gs, &B[i], ((double) rand() / (RAND_MAX + 1)));
    }
    for (int i = 0; i < widthC * heightC; i++) {
        ad_init_var(gs, &C[i], 0.01);
    }

    lastid = gs->current_variable_id;

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
    cl::Buffer c_d;

    error = CL_SUCCESS;
    std::string source_code;

    //Read the ad4cl api.
    std::string line;
    std::ifstream in;
    in.open("../../ad.cl");

    std::stringstream ss;

    while (in.good()) {
        std::getline(in, line);
        ss << line << "\n";
    }

    std::ifstream kin;
    kin.open("matrixmul.cl");

    while (kin.good()) {
        std::getline(kin, line);
        ss << line << "\n";
    }
    source_code = ss.str();

    std::vector<cl::Platform> platforms;

    try {
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            exit(0);
        }


        // Get list of devices on default platform and create context
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[1])(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
        devices = context.getInfo<CL_CONTEXT_DEVICES > ();




        //set the program source
        source = cl::Program::Sources(1, std::make_pair(source_code.c_str(), source_code.size()));
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
        kernel = cl::Kernel(program_, "matrixMult");

    } catch (cl::Error err) {
        std::cout << err.what() << "---> " << error << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG > (devices[0]);
    }

    gs_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof ( struct ad_gradient_structure), gs);
    ad_entry_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, GRADIENT_STACK_SIZE * sizeof (struct ad_entry), gradient_stack);
    a_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (widthA * heightA) * sizeof ( ad_variable), A);
    b_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, (widthB * heightB) * sizeof ( ad_variable), B);
    c_d = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, (widthC * heightC) * sizeof ( ad_variable), C);


    kernel.setArg(0, gs_d);
    kernel.setArg(1, ad_entry_d);
    kernel.setArg(2, a_d);
    kernel.setArg(3, b_d);
    kernel.setArg(4, c_d);
    kernel.setArg(5, widthA);
    kernel.setArg(6, widthB);
    for (int iter = 0; iter < 37; iter++) {
#ifdef HOST
#ifdef CL_PROFILING
        static struct timeval tm1, tm2;
        gettimeofday(&tm1, NULL);
#endif
        MatrixMultHost(gs, A, B, C);
#ifdef CL_PROFILING
        gettimeofday(&tm2, NULL);
        double t = 1000.00 * (double) (tm2.tv_sec - tm1.tv_sec) + (double) (tm2.tv_usec - tm1.tv_usec) / 1000.000;
        cout << "kernel equivalent time " << t << " ms" << std::endl;
#endif
#else

        //
        cl::Event event;
        try {
            queue.enqueueWriteBuffer(gs_d, CL_TRUE, 0, sizeof ( ad_gradient_structure), gs);
            queue.enqueueWriteBuffer(a_d, CL_TRUE, 0, (widthA * heightA) * sizeof ( ad_variable), A);
            queue.enqueueWriteBuffer(b_d, CL_TRUE, 0, (widthB * heightB) * sizeof ( ad_variable), B);



            queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(widthA, heightB),
                    cl::NDRange(16, 16),
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
            cout << "kernel time " << time << " ms" << std::endl;

#endif

            queue.enqueueReadBuffer(c_d, CL_TRUE, 0, (widthC * heightC) * sizeof ( ad_variable), C);



        } catch (cl::Error err) {
            std::cout << error << err.what() << event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
        }

#endif
        
        gs->stack_current = 0;
        gs->current_variable_id = lastid + 1;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] gradient_stack;
    delete gs;
    return 0;
}

