EXECUTABLE=simple

INCLUDES=-I/C/admb/include -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\include"

LDFLAGS= -L/C/admb/lib -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\Win32"

LIBS = -ladmb -lOpenCL
CC=mingw32-g++

CFLAGS=-O3 -fpermissive -Wall -static

SOURCES = ../simple.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): 
	$(CC) $(CFLAGS) $(LDFLAGS)  $(INCLUDES) $(SOURCES) $< -o $@  $(LIBS)

clean:
	rm simple.exe