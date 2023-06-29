#include "library_fixed.h"
#include <iostream>
using namespace std;

int party = 0;
int port = 32000;
string address = "127.0.0.1";
int num_threads = 1;
int32_t bitlength = 41;
int32_t kScale = 12;

void fill(intType* k, size_t cnt) {
  for (size_t i=0; i<cnt; i++) k[i] = i % 5;
}

int main(int argc, char **argv) {
  ArgMapping amap;

  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("nt", num_threads, "Number of Threads");
  amap.arg("ell", bitlength, "Uniform Bitwidth");
  amap.arg("k", kScale, "scaling factor");

  amap.parse(argc, argv);

  assert(party == SERVER || party == CLIENT);
  std::cerr << "Loading input from stdin..." << std::endl;
  StartComputation();

  
  size_t n = 1; size_t m = 256; size_t r = 100;
  assert(n == 1);
  auto A = new intType[m * n]; fill(A, m*n);
  auto B = new intType[m * r]; fill(B, m*r);
  auto C = new intType[n * r]; 

  MatMul2D(n, m, r, A, B, C, false);


  
  // size_t n = 1; size_t imsize = 16; size_t ci = 64; size_t co = 64;
  // size_t kersize = 5; size_t stride = 1;
  // assert(n == 1);
  // size_t input_size = n * ci * imsize * imsize;
  // size_t kernel_size = ci * co * kersize * kersize;
  // std::cout << "ksize " << kernel_size << std::endl;
  // size_t output_size = n * co * imsize * imsize;
  // auto A = new intType[input_size]; fill(A, input_size);
  // auto B = new intType[kernel_size]; fill(B, kernel_size);
  // auto C = new intType[output_size]; 
  
  // Conv2DWrapper(n, imsize, imsize, ci, kersize, kersize, co, 0, 0, 0, 0, stride, stride, A, B, C);

  EndComputation();
  finalize();

  return 0;

}