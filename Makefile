# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O2
endif

# Compiler
CXX := g++

# Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )

# Target
TARGET := host
ALTERA_TARGET_DIR := build/altera

# Directories
INC_DIRS := inc
LIB_DIRS := 

# Files
INCS := $(wildcard inc/*.h)
SRCS := $(wildcard src/host/*.cpp src/AOCLUtils/*.cpp)
LIBS := rt pthread

# altera : $(TARGET_DIR)/$(TARGET)

altera : Makefile $(SRCS) $(INCS) $(ALTERA_TARGET_DIR)
	$(ECHO)$(CXX) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) $(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
		$(foreach D,$(LIB_DIRS),-L$D) \
		$(foreach L,$(LIBS),-l$L) \
		-o $(ALTERA_TARGET_DIR)/$(TARGET) -g

$(ALTERA_TARGET_DIR) :
	$(ECHO)mkdir $(ALTERA_TARGET_DIR)

clean :
	$(ECHO)rm -f $(ALTERA_TARGET_DIR)/$(TARGET)
