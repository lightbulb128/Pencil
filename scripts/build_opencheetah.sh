cd OpenCheetah
git submodule update --init --recursive

# clone and build dependencies
bash scripts/build-deps.sh

# clone pybind11 for python utils
cd SCI/extern/
git clone https://github.com/pybind/pybind11.git
cd ../..

# cmake and build opencheetah
cd build
cmake ..
make

# return to top directory
cd ../..