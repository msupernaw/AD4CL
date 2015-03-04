# AD4CL
Automatic Differentiation for OpenCL.
Requires a GPU with fp64 extension

Status: Patiently waiting for for OpenCL 2.0 so we can
use shared virtual memory. This will significantly reduce
the time to transfer the gradient buffers from the host to the device.
 