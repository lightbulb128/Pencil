#include <vector>
#include <complex>
#include <iostream>
#include "SCIProvider.h"

#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include "../extern/pybind11/include/pybind11/complex.h"
#include "../extern/pybind11/include/pybind11/stl_bind.h"
#include "../extern/pybind11/include/pybind11/numpy.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

#if defined(BIT41)
    #ifdef PARTY_ALICE
        #define PACKAGE_NAME cheetah_provider_alice_41
    #else
        #define PACKAGE_NAME cheetah_provider_bob_41
    #endif
#elif defined(BIT37)
    #ifdef PARTY_ALICE
        #define PACKAGE_NAME cheetah_provider_alice_37
    #else
        #define PACKAGE_NAME cheetah_provider_bob_37
    #endif
#else
    #ifdef PARTY_ALICE
        #define PACKAGE_NAME cheetah_provider_alice
    #else
        #define PACKAGE_NAME cheetah_provider_bob
    #endif
#endif

std::vector<uint64_t> getVectorFromBuffer(py::array_t<uint64_t>& values) {
    py::buffer_info buf = values.request();
    uint64_t *ptr = (uint64_t *)buf.ptr;
    std::vector<uint64_t> vec(buf.shape[0]);
    for (auto i = 0; i < buf.shape[0]; i++)
        vec[i] = ptr[i];
    return vec;
}

py::array_t<uint64_t> getBufferFromVector(const std::vector<uint64_t>& vec) {
    py::array_t<uint64_t> values(vec.size());
    py::buffer_info buf = values.request();
    uint64_t *ptr = (uint64_t *)buf.ptr;
    for (auto i = 0; i < buf.shape[0]; i++)
        ptr[i] = vec[i];
    return values;
}

std::vector<uint8_t> getVectorFromBuffer(py::array_t<uint8_t>& values) {
    py::buffer_info buf = values.request();
    uint8_t *ptr = (uint8_t *)buf.ptr;
    std::vector<uint8_t> vec(buf.shape[0]);
    for (auto i = 0; i < buf.shape[0]; i++)
        vec[i] = ptr[i];
    return vec;
}

py::array_t<uint8_t> getBufferFromVector(const std::vector<uint8_t>& vec) {
    py::array_t<uint8_t> values(vec.size());
    py::buffer_info buf = values.request();
    uint8_t *ptr = (uint8_t *)buf.ptr;
    for (auto i = 0; i < buf.shape[0]; i++)
        ptr[i] = vec[i];
    return values;
}

PYBIND11_MODULE(PACKAGE_NAME, m) {

    py::class_<CheetahProvider>(m, "CheetahProvider")
        .def(py::init<int>())
        .def("startComputation", &CheetahProvider::startComputation)
        .def("endComputation", &CheetahProvider::endComputation)
        .def("dbits", &CheetahProvider::dbits)
        .def("relu", [](CheetahProvider& self, py::array_t<uint64_t> share, bool truncate) {
            auto p = self.relu(getVectorFromBuffer(share), truncate);
            return std::make_pair(getBufferFromVector(std::move(p.first)), getBufferFromVector(std::move(p.second)));
        })
        .def("drelumul", [](CheetahProvider& self, py::array_t<uint64_t> share, py::array_t<uint8_t> drelu) {
            auto ret = self.drelumul(getVectorFromBuffer(share), getVectorFromBuffer(drelu));
            return getBufferFromVector(std::move(ret));
        })
        .def("truncate", [](CheetahProvider& self, py::array_t<uint64_t> share) {
            auto ret = self.truncate(getVectorFromBuffer(share));
            return getBufferFromVector(std::move(ret));
        })
        .def("divide", [](CheetahProvider& self, py::array_t<uint64_t> share, uint64_t divisor) {
            auto ret = self.divide(getVectorFromBuffer(share), divisor);
            return getBufferFromVector(std::move(ret));
        })
        .def("max", [](CheetahProvider& self, py::array_t<uint64_t> share, size_t result_count) {
            auto ret = self.max(getVectorFromBuffer(share), result_count);
            return getBufferFromVector(std::move(ret));
        })
    ;
}