#include <assert.h>

#include <chrono>
#include <iostream>
#include <set>
#include <vector>

#include "hnswlib/hnswlib.h"

using idx_t = hnswlib::labeltype;

template <typename T>
void read_vecs(const std::string &filename, T *&data, int &nx, int &dim) {
  std::ifstream fs(filename.c_str(), std::ios::binary);
  fs.read((char *)&dim, 4);
  fs.seekg(0, std::ios::end);
  nx = fs.tellg() / (dim + 1) / 4;
  fs.seekg(0, std::ios::beg);
  std::cout << "Read path: " << filename << ", nx: " << nx << ", dim: " << dim
            << std::endl;
  data = new T[nx * dim];
  for (int i = 0; i < nx; ++i) {
    fs.seekg(4, std::ios::cur);
    fs.read((char *)&data[i * dim], dim * sizeof(T));
  }
}

namespace hy {

template <typename T = double> struct OnlineEV {
  static_assert(std::is_arithmetic_v<T>, "T should be arithmetic type");

  T s1 = 0, s2 = 0, n = 0;

  void insert(T x) {
    n++;
    s1 += x;
    s2 += x * x;
  }

  void erase(T x) {
    n--;
    s1 -= x;
    s2 -= x * x;
  }

  double E() {
    if (n == 0) {
      return -1;
    }
    return static_cast<double>(s1) / n;
  }

  double V() {
    if (n == 0) {
      return -1;
    }
    return static_cast<double>(s2) / n - static_cast<double>(s1 * s1) / n / n;
  }
};

} // namespace hy

void TEST() {
  int nx, nq, dim, K;
  float *query;
  int *gt;
  read_vecs<float>("../../tests/data/sift/sift_query.fvecs", query, nq, dim);
  read_vecs<int>("../../tests/data/sift/sift_groundtruth.ivecs", gt, nq, K);

  auto space = new hnswlib::L2Space(dim);

  auto hnsw = new hnswlib::HierarchicalNSW<float>(space, "hnswflat.index");

  hnsw->setEf(32);
  int64_t cnt = 0;
  int query_k = 10;
  hy::OnlineEV<double> ev;
  constexpr int iters = 10;
  for (int iter = 0; iter <= iters; ++iter) {
    auto st = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nq; ++i) {
      auto res = hnsw->searchKnnCloserFirst(&query[i * dim], query_k);
      std::unordered_set<int> st(&gt[i * K], &gt[i * K + query_k]);
      for (auto &[x, y] : res)
        if (st.count(y))
          cnt++;
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto tcnt =
        std::chrono::duration_cast<std::chrono::microseconds>(ed - st).count();
    if (iter == 0) {
      std::cout << "Recall: " << double(cnt) / nq / query_k << std::endl;
    } else {
      double qps = nq / (tcnt / 1000000.0);
      ev.insert(qps);
      std::cout << "QPS: " << qps << ", Mean: " << ev.E()
                << ", Variance: " << ev.V() << std::endl;
    }
  }
  delete space;
  delete hnsw;
}

int main() {
  std::cout << "Testing ..." << std::endl;
  TEST();
  std::cout << "Test ok" << std::endl;

  return 0;
}
