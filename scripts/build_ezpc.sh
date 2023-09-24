cd EzPC/SCI/extern

git clone https://github.com/pybind/pybind11.git

cp locks.h SEAL/native/src/seal/util/locks.h

cd ..

mkdir -p build
cd build
cmake ..
make
