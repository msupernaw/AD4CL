EXECUTABLE=matrix_mul

INCLUDES= -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\include"

LDFLAGS= -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\Win32"

LIBS = -lOpenCL
CC=mingw32-g++

CFLAGS=-O3 -fpermissive -Wall -static

SOURCES = matrix_mul.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): 
	$(CC) $(CFLAGS) $(LDFLAGS)  $(INCLUDES) $(SOURCES) $< -o $@  $(LIBS) -static

clean:
	rm matrix_mul.exe