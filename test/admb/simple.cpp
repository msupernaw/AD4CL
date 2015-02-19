
#include <admodel.h>
#include <cmath>
#define CL_PROFILING
#include <vector>
#include <sys/time.h>

extern "C" {
    void ad_boundf(int i);
}
#include "simple.hpp"

inline void AD(struct ad_gradient_structure* gs,
        struct ad_variable* a,
        struct ad_variable*b,
        double *x,
        double *y,
        struct ad_variable *out, int size) {

    for (int i = 0; i < size; i++) {
        //        struct ad_variable pred = ad_plus(gs, ad_times_vd(gs, *a, x[i]), *b);
        struct ad_variable temp = ad_minus_vd(gs, ad_plus(gs, ad_times_vd(gs, *a, x[i]), *b), y[i]);
        out[i] = ad_times(gs, temp, temp);
    }


}

void Gradient(std::vector<double>& g, struct ad_gradient_structure& gs) {

    std::fill(g.begin(), g.end(), 0.0);

    if (g.size() < gs.current_variable_id) {
        g.resize(gs.current_variable_id + 1);
    }
    g[gs.gradient_stack[gs.stack_current - 1].id] = 1.0;

    for (int j = gs.stack_current - 1; j >= 0; j--) {
        int id = gs.gradient_stack[j].id;
        double w = g[id];
        g[id] = 0.0;
        for (int i = 0; i < gs.gradient_stack[j].size; i++) {
            g[gs.gradient_stack[j].coeff[i].id] += w * gs.gradient_stack[j].coeff[i].dx;
        }
    }
}

model_data::model_data(int argc, char * argv[]) : ad_comm(argc, argv) {
    nobs.allocate("nobs");
    method.allocate("method");
    ad4cl_stack_size.allocate("ad4cl_stack_size");
    YY.allocate(1, nobs);
    XX.allocate(1, nobs);
    A = 2.0;
    B = 4.0;
    S = 7.0;
    random_number_generator rng(101);
    dvector err(1, nobs);
    XX.fill_randu(rng);
    XX *= 150.0;
    YY = A * XX + B;


    err.fill_randn(rng);
    YY += S*err;



    Y = new double[nobs.val];
    x = new double[nobs.val];



    std::cout << nobs.val << std::endl;

    for (int i = 0; i < nobs; i++) {

        x[i] = XX[i + 1];
        Y[i] = YY[i + 1];
    }




}

model_parameters::model_parameters(int sz, int argc, char * argv[]) :
model_data(argc, argv), function_minimizer(sz) {
    initializationfunction();
    gradient_method = method;
    a.allocate("a");
    //        a = .0000012;
    b.allocate("b");
    pred_Y.allocate(1, nobs, "pred_Y");
#ifndef NO_AD_INITIALIZE
    pred_Y.initialize();
#endif
    f.allocate("f");
    prior_function_value.allocate("prior_function_value");
    likelihood_function_value.allocate("likelihood_function_value");

    out = new ad_variable[nobs.val];

    this->initialize_opencl();

}

void model_parameters::initialize_opencl() {

    DATA_SIZE = nobs.val;


    gs = new ad_gradient_structure();
    gs->current_variable_id = 1;
    gs->stack_current = 0;
    gs->recording = 1;
    gs->counter = 1;

    this->gradient_stack = new ad_entry[this->ad4cl_stack_size.val];
    for (int i = 0; i < this->ad4cl_stack_size.val; i++) {
        this->gradient_stack[i].size = 0;
        this->gradient_stack[i].id = 0;
    }
    gs->gradient_stack = this->gradient_stack;

    aa = (struct ad_variable){.value = 0, .id = gs->current_variable_id++};
    bb = (struct ad_variable){.value = 0, .id = gs->current_variable_id++};
    out = new ad_variable[DATA_SIZE];



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
    kin.open("simple.cl");

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

        //print platform info 
        //        std::cout << platforms[1];

        // Get list of devices on default platform and create context
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[1])(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
        devices = context.getInfo<CL_CONTEXT_DEVICES > ();

        //        std::cout << __LINE__ << std::endl;
        const cl::Device device = devices[0];

        //print device info
        //        std::cout << device << "\n";

        //set the program source
        source = cl::Program::Sources(1, std::make_pair(source_code.c_str(), source_code.size()));
        program_ = cl::Program(context, source, &error);
        //        std::cout << __LINE__ << std::endl;
        //build the program
        program_.build(devices);
        //        std::cout << __LINE__ << std::endl;
        //set the queue
#ifdef CL_PROFILING
        queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
#else
        queue = cl::CommandQueue(context, devices[0]);
#endif
        //        std::cout << __LINE__ << std::endl;
        if (error != CL_SUCCESS) {
            std::cout << "---> " << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG > (devices[0]) << "\n";
            exit(0);
        }
        //        std::cout << __LINE__ << std::endl;

        // Create kernel object
        kernel = cl::Kernel(program_, "AD");

        //std::cout<<"here"<<std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "---> " << error << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG > (devices[0]);
    }


    gs_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof ( struct ad_gradient_structure), gs);
    ad_entry_d = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, this->ad4cl_stack_size.val * sizeof (struct ad_entry), gradient_stack);
    a_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof ( ad_variable), &aa);
    b_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof ( ad_variable), &bb);
    x_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, DATA_SIZE * sizeof (double), x);
    y_d = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, DATA_SIZE * sizeof (double), Y);
    out_d = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, DATA_SIZE * sizeof (struct ad_variable), out);


    // Number of work items in each local work group
    local_size = 64; //devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE > ();

    // Number of total work items - localSize must be devisor
    global_size = std::ceil(DATA_SIZE / (double) local_size + 1) * local_size;



    //    queue.enqueueWriteBuffer(x_d, CL_TRUE, 0, sizeof (double)*DATA_SIZE, x);
    //    queue.enqueueWriteBuffer(y_d, CL_TRUE, 0, sizeof (double)*DATA_SIZE, Y);

    kernel.setArg(0, gs_d);
    kernel.setArg(1, ad_entry_d);
    kernel.setArg(2, a_d);
    kernel.setArg(3, b_d);
    kernel.setArg(4, x_d);
    kernel.setArg(5, y_d);
    kernel.setArg(6, out_d);
    kernel.setArg(7, DATA_SIZE);


}

void model_parameters::userfunction(void) {

#ifdef CL_PROFILING
    static struct timeval utm1, utm2;
    gettimeofday(&utm1, NULL);
#endif

    f = 0.0;
    aa.value = a.xval();
    bb.value = b.xval();

    if (gradient_method == AD4CL_DEVICE) {
        try {
            queue.enqueueWriteBuffer(gs_d, CL_TRUE, 0, sizeof ( ad_gradient_structure), gs);
            queue.enqueueWriteBuffer(a_d, CL_TRUE, 0, sizeof ( ad_variable), &aa);
            queue.enqueueWriteBuffer(b_d, CL_TRUE, 0, sizeof ( ad_variable), &bb);


            cl::Event event;
            queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(global_size),
                    cl::NDRange(local_size),
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
            cout << "kernel time " << time << " ms, ";
            ;
#endif
            queue.enqueueReadBuffer(ad_entry_d, CL_TRUE, 0, this->ad4cl_stack_size.val * sizeof ( ad_entry), (struct ad_entry*) gradient_stack);

            queue.enqueueReadBuffer(out_d, CL_TRUE, 0, DATA_SIZE * sizeof ( ad_variable), (struct ad_variable*) out);

            queue.enqueueReadBuffer(gs_d, CL_TRUE, 0, sizeof ( ad_gradient_structure), gs);

            //         exit(0);
            gs->gradient_stack = gradient_stack;

            gpu_restore(gs);


            ad_variable sum = {.value = 0.0, .id = gs->current_variable_id++};
            for (int i = 0; i < DATA_SIZE; i++) {
                ad_plus_eq_v(gs, &sum, out[i]);
                out[i].value = 0;
            }


            //finish up with the native api.
            struct ad_variable ff = ad_times_dv(gs, static_cast<double> (DATA_SIZE) / 2.0, ad_log(gs, sum));


            //compute gradient
            Gradient(gradient, *gs);


            //set admb adjoint code
            f.v->xvalue() = ff.value;
            AD_SET_DERIVATIVES2(f, a, gradient[aa.id], b, gradient[bb.id]);

        } catch (cl::Error err) {
            std::cout << err.what() << std::endl;
            std::cout << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG > (devices[0]);
        }

        //reset the ad4cl gradient structure
        gs->stack_current = 0;
        gs->current_variable_id = bb.id + 1;

    } else if (gradient_method == AD4CL_HOST) {

#ifdef CL_PROFILING
        static struct timeval tm1, tm2;
        gettimeofday(&tm1, NULL);
#endif
        AD(gs, &aa, &bb, x, Y, out, DATA_SIZE);

#ifdef CL_PROFILING
        gettimeofday(&tm2, NULL);
        double t = 1000.00 * (double) (tm2.tv_sec - tm1.tv_sec) + (double) (tm2.tv_usec - tm1.tv_usec) / 1000.000;
        cout << "kernel equivalent time " << t << " ms, ";
#endif
        ad_variable sum = {.value = 0.0, .id = gs->current_variable_id++};
        for (int i = 0; i < DATA_SIZE; i++) {
            ad_plus_eq_v(gs, &sum, out[i]);
            out[i].value = 0;
        }


        //finish up with the native api.
        struct ad_variable ff = ad_times_dv(gs, static_cast<double> (DATA_SIZE) / 2.0, ad_log(gs, sum));

        //compute the gradient
        Gradient(gradient, *gs);

        //set the admb adjoint code
        f.v->xvalue() = ff.value;
        AD_SET_DERIVATIVES2(f, a, gradient[aa.id], b, gradient[bb.id]);

        //reset the ad4cl gradient structure
        gs->stack_current = 0;
        gs->current_variable_id = bb.id + 1;

    } else {

//#ifdef CL_PROFILING
//        static struct timeval tm1, tm2;
//        gettimeofday(&tm1, NULL);
//#endif
        pred_Y = (a * XX + b) - YY;


//#ifdef CL_PROFILING
//        gettimeofday(&tm2, NULL);
//        double t = 1000.00 * (double) (tm2.tv_sec - tm1.tv_sec) + (double) (tm2.tv_usec - tm1.tv_usec) / 1000.000;
//          cout << "kernel equivalent time " << time << " ms, ";
//#endif
        f = (norm2(pred_Y));
        f = nobs / 2. * log(f);


    }

#ifdef CL_PROFILING
    gettimeofday(&utm2, NULL);
    double t = 1000.00 * (double) (utm2.tv_sec - utm1.tv_sec) + (double) (utm2.tv_usec - utm1.tv_usec) / 1000.000;
    std::cout << "user function time: " << t << " ms" << std::endl;
#endif
}

void model_parameters::preliminary_calculations(void) {
#if defined(USE_ADPVM)

    admaster_slave_variable_interface(*this);

#endif
}

model_data::~model_data() {
}

model_parameters::~model_parameters() {
}

void model_parameters::report(const dvector & gradients) {
}

void model_parameters::final_calcs(void) {
}

void model_parameters::set_runtime(void) {
}

#ifdef _BORLANDC_
extern unsigned _stklen = 10000U;
#endif


#ifdef __ZTC__
extern unsigned int _stack = 10000U;
#endif

long int arrmblsize = 0;

int main(int argc, char * argv[]) {
    ad_set_new_handler();
    ad_exit = &ad_boundf;
    gradient_structure::set_NO_DERIVATIVES();
    gradient_structure::set_YES_SAVE_VARIABLES_VALUES();
    if (!arrmblsize) arrmblsize = 65000000;
    model_parameters mp(arrmblsize, argc, argv);
    mp.iprint = 10;
    mp.preliminary_calculations();
    mp.computations(argc, argv);
    return 0;
}

extern "C" {

    void ad_boundf(int i) {
        /* so we can stop here */
        exit(i);
    }
}
