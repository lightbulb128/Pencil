cd OpenCheetah
git submodule update --init --recursive

# clone and build dependencies
bash scripts/build-deps.sh

# clone pybind11 for python utils
cd SCI/extern/
git clone https://github.com/pybind/pybind11.git
cd ../..

# cmake and build opencheetah
bash scripts/build.sh

# return to top directory
cd ../..