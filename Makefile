# Makefile with Automatic Compiler Detection (nvcc/g++)
#
# This Makefile checks for the presence of the NVIDIA CUDA Compiler (nvcc).
# - If nvcc is found, it compiles the project with CUDA support.
# - If nvcc is not found, it falls back to using g++.

# --- Configuration ---

# Target executable name and location.
TARGET_DIR = $(HOME)
TARGET_NAME = cnet
TARGET = $(TARGET_DIR)/$(TARGET_NAME)

# Directory for build artifacts (object files).
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# --- Compiler Auto-Detection ---

# Check if the 'nvcc' command exists in the system's PATH.
NVCC_RESULT := $(shell which nvcc 2>/dev/null)
NVCC_TEST := $(notdir $(NVCC_RESULT))

# --- Conditional Build Configuration ---

ifeq ($(NVCC_TEST),nvcc)
    # --- NVCC Configuration (CUDA Build) ---
    INFO_MSG = "==> NVCC detected. Building with CUDA support."
    CXX = nvcc
    # Flags for nvcc:
    # -x cu: Treat all source files as CUDA files.
    # -Xcompiler -fopenmp: Pass the -fopenmp flag to the host compiler (like g++).
    CXXFLAGS = -x cu -std=c++11 -Xcompiler -fopenmp -O2
    LDFLAGS = -lgomp # Link with the GNU OpenMP runtime library.
    SRCS = utils/flags.cpp \
           gpu/gpu_func.cpp \
           impl/vars.cpp \
           gpu/kernels.cpp \
           gpu/reduce.cpp \
           gpu/reducegrad.cpp \
           cnet.cpp

else
    # --- G++ Configuration (Standard Build) ---
    INFO_MSG = "==> NVCC not found. Building with g++."
    CXX = g++
    # Note: -Xclang is specific to the Clang frontend. -fopenmp is the standard g++ flag.
    CXXFLAGS = -std=c++11 -Xclang -fopenmp -O2
    LDFLAGS = -lomp # Link with the OpenMP library.
    SRCS = utils/flags.cpp \
           impl/vars.cpp \
           gpu/gpu_func.cpp \
           gpu/reduce.cpp \
           gpu/reducegrad.cpp \
           gpu/kernels.cpp \
           cnet.cpp
endif

# --- Automatic Variable Generation ---

# VPATH tells 'make' where to look for source files.
# This is created automatically from the unique directory paths in the SRCS variable.
VPATH = $(sort $(dir $(SRCS)))

# Generate object file names, placing them in the OBJ_DIR.
# This handles both .cpp and .cu files correctly.
# e.g., gpu/reduce.cpp becomes build/obj/reduce.o
# e.g., gpu/gpu_func.cu becomes build/obj/gpu_func.o
OBJS = $(addprefix $(OBJ_DIR)/, $(notdir $(SRCS:.cpp=.o)))
OBJS := $(subst .cu,.o,$(OBJS))


# --- Build Rules ---

# Phony targets are not actual files. 'all' is the default goal.
.PHONY: all clean run info

# Default target: build the final executable.
all: $(TARGET)

# Rule to link the final executable from all the object files.
$(TARGET): $(OBJS)
	@echo "==> Linking target: $@"
	@mkdir -p $(TARGET_DIR) # Ensure the target directory exists.
	$(CXX) $^ $(LDFLAGS) -o $@
	@echo "==> Build complete. Executable is at $(TARGET)"

# Generic pattern rule to compile any source file (.cpp or .cu) into an object file.
# This works because both nvcc (with -x cu) and g++ can handle .cpp files.
# The VPATH variable allows 'make' to find the source file in its original directory.
$(OBJ_DIR)/%.o: %.cpp
	@echo "==> Compiling $<..."
	@mkdir -p $(OBJ_DIR) # Ensure the object directory exists.
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	@echo "==> Compiling $<..."
	@mkdir -p $(OBJ_DIR) # Ensure the object directory exists.
	$(CXX) $(CXXFLAGS) -c $< -o $@


# --- Utility Rules ---

# Rule to display which compiler is being used.
info:
	@echo $(INFO_MSG)

# Rule to run the compiled application.
run: all
	@echo "==> Running application..."
	@$(TARGET)
	@echo "==> Application finished."

# Rule to clean up all generated files.
clean:
	@echo "==> Cleaning up build artifacts..."
	@rm -rf $(BUILD_DIR) $(TARGET)
	@echo "==> Cleanup complete."


