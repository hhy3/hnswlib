#include <assert.h>

#include <iostream>
#include <set>
#include <vector>

#include "hnswlib/hnswlib.h"

class StopW {
  std::chrono::high_resolution_clock::time_point time_begin;

 public:
  StopW() { time_begin = std::chrono::high_resolution_clock::now(); }

  float getElapsedTimeNano() {
    std::chrono::high_resolution_clock::time_point time_end =
        std::chrono::high_resolution_clock::now();
    return (std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                                 time_begin)
                .count());
  }

  void reset() { time_begin = std::chrono::high_resolution_clock::now(); }
};

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
  float *query;
  int* gt;
  read_vecs<float>("../../tests/data/siftsmall/siftsmall_query.fvecs", query,
                   nq, dim);
  read_vecs<int>("../../tests/data/siftsmall/siftsmall_groundtruth.ivecs", gt,
                 nq, K);

  auto space = new hnswlib::L2Space(dim);

  auto hnsw = new hnswlib::HierarchicalNSW<float>(space, "hnsw2.index");

  hnsw->getCodes();
  hnsw->setEf(64);
  int64_t cnt = 0;
  StopW timer;
  double t = 0.0;
  int query_k = 10;
  for (int i = 0; i < nq; ++i) {
    timer.reset();
    auto res = hnsw->searchKnnCloserFirst(&query[i * dim], query_k);
    t += timer.getElapsedTimeNano();
    std::set<int> st(&gt[i * K], &gt[i * K + query_k]);
    for (auto& [x, y] : res)
      if (st.count(y)) cnt++;
  }
  std::cout << "Recall: " << double(cnt) / nq / query_k << std::endl;
  std::cout << "Latency: " << t / nq / 1000000 << "ms" << std::endl;
  std::cout << "ADC: " << (double)hnsw->T / cnt << "ns" << std::endl;
  delete space;
  delete hnsw;
}

int main() {
  std::cout << "Testing ..." << std::endl;
  TEST();
  std::cout << "Test ok" << std::endl;

  return 0;
}
