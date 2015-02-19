#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=mingw32-g++
CXX=mingw32-g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=MinGW-Windows
CND_DLIB_EXT=dll
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/simple.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=../../../ad4cl_with_admb/admb-master/build/dist/lib/libadmb.a ../../../../../../../NVIDIA/CUDA/CUDAToolkit/lib/Win32/OpenCL.lib

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/simple.exe

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/simple.exe: ../../../ad4cl_with_admb/admb-master/build/dist/lib/libadmb.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/simple.exe: ../../../../../../../NVIDIA/CUDA/CUDAToolkit/lib/Win32/OpenCL.lib

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/simple.exe: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/simple ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/simple.o: simple.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../../../ad4cl_with_admb/admb-master/build/dist/include -I../../../../../../../NVIDIA/CUDA/CUDAToolkit/include -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/simple.o simple.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/simple.exe

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
