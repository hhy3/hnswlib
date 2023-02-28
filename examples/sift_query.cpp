#include <assert.h>
#include <bits/stdc++.h>

#include <chrono>
#include <iostream>
#include <set>
#include <vector>

#include "hnswlib/hnswlib.h"

using namespace std;

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
template <typename T = double>
struct OnlineEV {
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

}  // namespace hy

constexpr int topk = 10, iters = 1;

void TEST() {
  int n, nq, dim, gt_k;
  float *base;
  float *query;
  int *gt;
  read_vecs<float>("../../tests/data/sift/sift_base.fvecs", base, n, dim);
  read_vecs<float>("../../tests/data/sift/sift_query.fvecs", query, nq, dim);
  read_vecs<int>("../../tests/data/sift/sift_groundtruth.ivecs", gt, nq, gt_k);

  auto space = new hnswlib::InnerProductSpace(dim);

  int cnt = 0;
  cout << "BEFORE" << endl;
  auto hnsw = new hnswlib::HierarchicalNSW<float>(space, "hnsw_fuck.index");
  cout << "AFTER" << endl;
  auto ep = hnsw->enterpoint_node_;
  queue<int> q;
  vector<bool> vis(n);
  vis[ep] = true;
  q.push(ep);
  int total = 1;
  while (q.size()) {
    int u = q.front();
    q.pop();
    int* list = (int*)hnsw->get_linklist0(u);
    int m = list[0];
    for (int j = 1; j <= m; ++j) {
      int v = list[j];
      if (!vis[v]) {
        q.push(v);
        vis[v] = true;
        total++;
      }
    }
  }
  cout << "FUCK " << total << endl;
  // vector<int> efs = {16, 24, 30, 36, 48, 64, 96, 128};
  // for (auto ef : efs) {
  //   hy::OnlineEV<double> ev;
  //   hnsw->setEf(ef);
  //   for (int iter = 0; iter <= iters; ++iter) {
  //     int64_t cnt = 0;
  //     auto st = std::chrono::high_resolution_clock::now();
  //     // #pragma omp parallel for
  //     for (int i = 0; i < nq; ++i) {
  //       auto res = hnsw->searchKnnCloserFirst(query + i * dim, topk);
  //       // std::unordered_set<int> st;
  //       // for (int j = 0; j < topk; ++j) {
  //       //   st.insert(hnsw->getExternalLabel(gt[i * gt_k + j]));
  //       // }
  //       std::unordered_set<int> st(gt + i * gt_k, gt + i * gt_k + topk);
  //       for (int j = 0; j < res.size(); ++j) {
  //         auto [dist, id] = res[j];
  //         // std::cout << j << " " << dist << " " << id << " " << gt[i * gt_k + j]
  //         //           << " " << hnsw->getExternalLabel(gt[i * gt_k + j])
  //         //           << std::endl;
  //         if (st.count(id)) {
  //           cnt++;
  //         }
  //       }
  //     }
  //     auto ed = std::chrono::high_resolution_clock::now();
  //     auto elapsed =
  //         std::chrono::duration_cast<std::chrono::microseconds>(ed - st)
  //             .count() /
  //         1000000.0;
  //     if (iter == 0) {
  //       double recall = double(cnt) / nq / topk;
  //       // writer.write((char *)&recall, sizeof(recall));
  //       std::cout << "ef: " << ef << "\n";
  //       std::cout << "  recall: " << recall << "\n";
  //     } else {
  //       auto qps = nq / elapsed;
  //       ev.insert(qps);
  //       // std::cout << ev.E() << std::endl;
  //     }
  //   }
  //   double qps = ev.E();
  //   std::cout << "  qps: " << qps << "\n";
  //   // writer.write((char *)&qps, sizeof(qps));
  // }
  delete space;
  delete hnsw;
}

int main() {
  std::cout << "Testing ..." << std::endl;
  TEST();
  std::cout << "Test ok" << std::endl;

  return 0;
}
