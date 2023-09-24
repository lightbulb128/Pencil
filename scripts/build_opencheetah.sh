# clone 
git clone https://github.com/lightbulb128/OpenCheetah.git
cd OpenCheetah
git checkout 43f0064ec71b6714a298f4acdc74dcf7fa99a8f1

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