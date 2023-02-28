#include <assert.h>

#include <iostream>
#include <set>
#include <vector>

#include "hnswlib/hnswlib.h"

using idx_t = hnswlib::labeltype;

template <typename T>
void read_vecs(const std::string& filename, T*& data, int& nx, int& dim) {
  std::ifstream fs(filename.c_str(), std::ios::binary);
  fs.read((char*)&dim, 4);
  fs.seekg(0, std::ios::end);
  nx = fs.tellg() / (dim + 1) / 4;
  fs.seekg(0, std::ios::beg);
  std::cout << "Read path: " << filename << ", nx: " << nx << ", dim: " << dim
            << std::endl;
  data = new T[nx * dim];
  for (int i = 0; i < nx; ++i) {
    fs.seekg(4, std::ios::cur);
    fs.read((char*)&data[i * dim], dim * sizeof(T));
  }
}

void TEST() {
  int nx, nq, dim, K;
  float* base;
  read_vecs<float>("../../tests/data/sift/sift_base.fvecs", base, nx, dim);

  auto space = new hnswlib::InnerProductSpace(dim);
  auto hnsw = new hnswlib::HierarchicalNSW<float>(space, nx, 1, 96);
  hnsw->build(base, nx, dim);
  //   hnsw->addPoint(&base[0], 0);
  // #pragma omp parallel for schedule(dynamic)
  //   for (int i = 1; i < nx; ++i) {
  //     hnsw->addPoint(&base[i * dim], i);
  //   }
  hnsw->saveIndex("hnsw_fuck.index");
  delete space;
  delete hnsw;
}

int main() {
  std::cout << "Testing ..." << std::endl;
  TEST();
  std::cout << "Test ok" << std::endl;

  return 0;
}
