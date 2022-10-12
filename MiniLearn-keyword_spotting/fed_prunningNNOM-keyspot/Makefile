OPENCV=0
OPENMP=0
DEBUG=0

OBJ=main.o image.o args.o matrix.o list.o data.o classifier.o net.o convolutional_layer.o connected_layer.o activations.o activation_layer.o arm_fully_connected_q7_opt.o\
 arm_convolve_HWC_q7_basic.o arm_convolve_HWC_q7_fast.o arm_relu_q7.o arm_pool_q7_HWC.o arm_convolve_HWC_q7_RGB.o arm_softmax_q7.o arm_max_q7.o arm_add_q7.o\
 arm_convolve_1x1_HWC_q7_fast_nonsquare.o arm_convolve_HWC_q15_fast.o arm_convolve_HWC_q15_fast_nonsquare.o arm_convolve_HWC_q7_basic_nonsquare.o\
 arm_convolve_HWC_q7_fast_nonsquare.o arm_depthwise_separable_conv_HWC_q7_nonsquare.o arm_sub_q7.o\
 arm_fully_connected_mat_q7_vec_q15_opt.o arm_mult_q7.o arm_nn_activations_q15.o arm_nn_activations_q7.o arm_relu_q15.o arm_nntables.o\

# EXOBJ=test.o

#CMSIS PATH
# CMSIS_DIR := /Users/christosprofentzas/CMSIS-ML/CMSIS_5

# INC=$($(CMSIS_DIR)/CMSIS/DSP/Include/) 
# INC_PARAMS=$(foreach d, $(INC), -I$d)


# INC_PARAMS = -I$(CMSIS_DIR)/CMSIS/DSP/Include/

INC=-I/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/DSP/Include/ \
	-I/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/Core/Include \
	-I/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Include \
	-I/Users/christosprofentzas/CMSIS-ML/nrf52840/00microTL/fed_prunningNNOM-keyspot/inc/ \
	-I/Users/christosprofentzas/CMSIS-ML/nrf52840/00microTL/fed_prunningNNOM-keyspot/inc/ \


# nnomcore = /Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/core/
# SRCDIR = /Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/layers/
# nnomback = /Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/backends/
# src = $(wildcard *.c)
# obj = $(src:.c=.o)


# OBJ += $(nnomcore %.c,%.o,$(wildcard *.c))
# OBJ += $(nnomlayers %.c,%.o,$(wildcard *.c))
# OBJ1 = $(nnomback %.c,%.o,$(wildcard *.c))

# OBJ1 = $(addprefix $(nnomback)/, $(patsubst %.c,%.o,$(wildcard *.c))  )


# OBJECTS     := $(patsubst , $(nnomback).c,%.o,$(wildcard *.c)) 
SRCEXT      := c
OBJEXT      := o
SRCCORE = /Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/core
SRCLAYERS = /Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/layers
SRCBACKEND = /Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/backends

LAYERSOURCES     := $(shell find $(SRCLAYERS) -type f -name *.$(SRCEXT))
OBJ     += $(patsubst $(SRCLAYERS)/%,$(obj)%,$(LAYERSOURCES:.$(SRCEXT)=.$(OBJEXT)))

CORESOURCES     := $(shell find $(SRCCORE) -type f -name *.$(SRCEXT))
OBJ     += $(patsubst $(SRCCORE)/%,$(obj)%,$(CORESOURCES:.$(SRCEXT)=.$(OBJEXT)))

BACKENDSOURCES     := $(shell find $(SRCBACKEND) -type f -name *.$(SRCEXT))
OBJ     += $(patsubst $(SRCBACKEND)/%,$(obj)%,$(BACKENDSOURCES:.$(SRCEXT)=.$(OBJEXT)))


# OBJECTS     := $(patsubst $(nnomback)/%, $(obj)/% , $(SOURCES:.$(SRCEXT)=.$(OBJEXT)))


$(info    OBJECTS are $(OBJ))
#   $(CMSIS_DIR)/CMSIS/DSP/Include \
#   $(CMSIS_DIR)/CMSIS/NN/Include \
# Include folders for nn of CMSIS
# INC_FOLDERS += \
#    \
#    \
#    \


# HEADER = $(CMSIS_DIR)/CMSIS/DSP/Include/arm_math.h






VPATH=./src/:./::/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Source/ConvolutionFunctions/\
	:/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Source/FullyConnectedFunctions/\
	:/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Source/ActivationFunctions/\
	:/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Source/SoftmaxFunctions/\
	:/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Source/NNSupportFunctions\
	:/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/NN/Source/PoolingFunctions/\
	:/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/DSP/Source/StatisticsFunctions/\
	:/Users/christosprofentzas/CMSIS-ML/CMSIS-updated/CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/\
	:/Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/layers/\
	:/Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/core/\
	:/Users/christosprofentzas/CMSIS-ML/nrf52840/nnom/src/backends/\


EXEC=uwnet
SLIB=lib${EXEC}.so
ALIB=lib${EXEC}.a
OBJDIR=./obj/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-O2
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/ 
CFLAGS=$(INC) -Wall -Wextra -Wno-unknown-pragmas -Wshift-overflow -ggdb3 -fPIC

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
COMMON= -Iinclude/ -Isrc/ 
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

EXOBJS = $(addprefix $(OBJDIR), $(EXOBJ))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile 

all: obj $(SLIB) $(ALIB) $(EXEC)

$(EXEC): $(EXOBJS) $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) 

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXOBJS) $(OBJDIR)

