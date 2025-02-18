CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC
PYTHON_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_EXTENSION_SUFFIX = $(shell python3-config --extension-suffix)
SOURCE_DIR = csrc
SOURCE_FILE = sequence_module.cpp
BUILD_DIR = build
LIB_DIR = $(BUILD_DIR)/lib
OUTPUT_FILE = $(LIB_DIR)/sequence_module$(PYTHON_EXTENSION_SUFFIX)

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(SOURCE_DIR)/$(SOURCE_FILE)
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDES) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: clean