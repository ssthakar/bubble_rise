# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kazeshini/ctorch/Hyesing_benchmark

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kazeshini/ctorch/Hyesing_benchmark/build

# Include any dependencies generated for this target.
include CMakeFiles/torch_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/torch_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/torch_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torch_test.dir/flags.make

CMakeFiles/torch_test.dir/main.cpp.o: CMakeFiles/torch_test.dir/flags.make
CMakeFiles/torch_test.dir/main.cpp.o: ../main.cpp
CMakeFiles/torch_test.dir/main.cpp.o: CMakeFiles/torch_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kazeshini/ctorch/Hyesing_benchmark/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torch_test.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torch_test.dir/main.cpp.o -MF CMakeFiles/torch_test.dir/main.cpp.o.d -o CMakeFiles/torch_test.dir/main.cpp.o -c /home/kazeshini/ctorch/Hyesing_benchmark/main.cpp

CMakeFiles/torch_test.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torch_test.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kazeshini/ctorch/Hyesing_benchmark/main.cpp > CMakeFiles/torch_test.dir/main.cpp.i

CMakeFiles/torch_test.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torch_test.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kazeshini/ctorch/Hyesing_benchmark/main.cpp -o CMakeFiles/torch_test.dir/main.cpp.s

CMakeFiles/torch_test.dir/nn_main.cpp.o: CMakeFiles/torch_test.dir/flags.make
CMakeFiles/torch_test.dir/nn_main.cpp.o: ../nn_main.cpp
CMakeFiles/torch_test.dir/nn_main.cpp.o: CMakeFiles/torch_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kazeshini/ctorch/Hyesing_benchmark/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/torch_test.dir/nn_main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torch_test.dir/nn_main.cpp.o -MF CMakeFiles/torch_test.dir/nn_main.cpp.o.d -o CMakeFiles/torch_test.dir/nn_main.cpp.o -c /home/kazeshini/ctorch/Hyesing_benchmark/nn_main.cpp

CMakeFiles/torch_test.dir/nn_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torch_test.dir/nn_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kazeshini/ctorch/Hyesing_benchmark/nn_main.cpp > CMakeFiles/torch_test.dir/nn_main.cpp.i

CMakeFiles/torch_test.dir/nn_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torch_test.dir/nn_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kazeshini/ctorch/Hyesing_benchmark/nn_main.cpp -o CMakeFiles/torch_test.dir/nn_main.cpp.s

# Object files for target torch_test
torch_test_OBJECTS = \
"CMakeFiles/torch_test.dir/main.cpp.o" \
"CMakeFiles/torch_test.dir/nn_main.cpp.o"

# External object files for target torch_test
torch_test_EXTERNAL_OBJECTS =

torch_test: CMakeFiles/torch_test.dir/main.cpp.o
torch_test: CMakeFiles/torch_test.dir/nn_main.cpp.o
torch_test: CMakeFiles/torch_test.dir/build.make
torch_test: /home/kazeshini/Ctorch/libtorch/lib/libtorch.so
torch_test: /home/kazeshini/Ctorch/libtorch/lib/libc10.so
torch_test: /home/kazeshini/Ctorch/libtorch/lib/libkineto.a
torch_test: /usr/lib/x86_64-linux-gnu/libcuda.so
torch_test: /usr/local/cuda-12.3/lib64/libnvrtc.so
torch_test: /usr/local/cuda-12.3/lib64/libnvToolsExt.so
torch_test: /usr/local/cuda-12.3/lib64/libcudart.so
torch_test: /home/kazeshini/Ctorch/libtorch/lib/libc10_cuda.so
torch_test: /home/kazeshini/Ctorch/libtorch/lib/libc10_cuda.so
torch_test: /home/kazeshini/Ctorch/libtorch/lib/libc10.so
torch_test: /usr/local/cuda-12.3/lib64/libcudart.so
torch_test: /usr/local/cuda-12.3/lib64/libnvToolsExt.so
torch_test: /usr/local/cuda-12.3/lib64/libcufft.so
torch_test: /usr/local/cuda-12.3/lib64/libcurand.so
torch_test: /usr/local/cuda-12.3/lib64/libcublas.so
torch_test: /usr/local/cuda-12.3/lib64/libcublasLt.so
torch_test: CMakeFiles/torch_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kazeshini/ctorch/Hyesing_benchmark/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable torch_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torch_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torch_test.dir/build: torch_test
.PHONY : CMakeFiles/torch_test.dir/build

CMakeFiles/torch_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torch_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torch_test.dir/clean

CMakeFiles/torch_test.dir/depend:
	cd /home/kazeshini/ctorch/Hyesing_benchmark/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kazeshini/ctorch/Hyesing_benchmark /home/kazeshini/ctorch/Hyesing_benchmark /home/kazeshini/ctorch/Hyesing_benchmark/build /home/kazeshini/ctorch/Hyesing_benchmark/build /home/kazeshini/ctorch/Hyesing_benchmark/build/CMakeFiles/torch_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torch_test.dir/depend

