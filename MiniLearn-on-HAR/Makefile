OPENCV=0
OPENMP=0
DEBUG=0

OBJ=main.o image.o args.o  matrix.o list.o data.o classifier.o net.o  maxpool_layer.o convolutional_layer.o connected_layer.o activations.o arm_fully_connected_q7.o\
 arm_convolve_HWC_q7_basic.o arm_convolve_HWC_q7_fast.o arm_relu_q7.o arm_pool_q7_HWC.o arm_convolve_HWC_q7_RGB.o arm_softmax_q7.o arm_max_q7.o batchnorm_layer.o activation_layer.o 
# EXOBJ=test.o

#CMSIS PATH
# CMSIS_DIR := /Users/christosprofentzas/CMSIS-ML/CMSIS_5
# INC=$($(CMSIS_DIR)/CMSIS/DSP/Include/) 
# INC_PARAMS=$(foreach d, $(INC), -I$d)
# INC_PARAMS = -I$(CMSIS_DIR)/CMSIS/DSP/Include/

INC=-I../CMSIS/CMSIS/DSP/Include/ \
	-I../CMSIS/CMSIS/Core/Include \
	-I../CMSIS/CMSIS/NN/Include \
# HEADER = $(CMSIS_DIR)/CMSIS/DSP/Include/arm_math.h


VPATH=./src/:./:../CMSIS/CMSIS/NN/Source/ConvolutionFunctions/\
	:../CMSIS/CMSIS/NN/Source/FullyConnectedFunctions/\
	:../CMSIS/CMSIS/Source/ActivationFunctions/\
	:../CMSIS/CMSIS/CMSIS/NN/Source/SoftmaxFunctions/\
	:../CMSIS/CMSIS/NN/Source/Source/Source/NNSupportFunction\
	:../CMSIS/CMSIS/NN/Source/PoolingFunctions/\
	:../CMSIS/CMSIS/DSP/Source/StatisticsFunctions/


EXEC=MiniLearn
SLIB=lib${EXEC}.so
ALIB=lib${EXEC}.a
OBJDIR=./obj/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-O2 -g
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/ 
CFLAGS=$(INC) -Wall -Wextra -Wno-unknown-pragmas -Wshift-overflow -ggdb3 -fPIC -g

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

