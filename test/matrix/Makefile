EXECUTABLE=matrix_mul

INCLUDES= -I/C/NVIDIA/CUDA/CUDAToolkit/include

LDFLAGS= -L/C/NVIDIA/CUDA/CUDAToolkit/lib/Win32 

LIBS =  -lOpenCL
CC=mingw32-g++

CFLAGS=-O3 -fpermissive -Wall 

SOURCES = matrix_mul.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): 
	$(CC) $(CFLAGS) $(LDFLAGS)  $(INCLUDES) $(SOURCES) $< -o $@  $(LIBS)

clean:
	rm matrix_mul.exe