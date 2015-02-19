EXECUTABLE=simple

INCLUDES=-I/C/Users/Matthew/Documents/NetBeansProjects/ad4cl_with_admb/admb-master/build/dist/include   -I/C/NVIDIA/CUDA/CUDAToolkit/include

LDFLAGS= -L/C/Users/Matthew/Documents/NetBeansProjects/ad4cl_with_admb/admb-master/build/dist/lib -L/C/NVIDIA/CUDA/CUDAToolkit/lib/Win32 

LIBS = -ladmb -lOpenCL
CC=mingw32-g++

CFLAGS=-O3 -fpermissive -Wall 

SOURCES = ../simple.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): 
	$(CC) $(CFLAGS) $(LDFLAGS)  $(INCLUDES) $(SOURCES) $< -o $@  $(LIBS)

clean:
	rm simple.exe