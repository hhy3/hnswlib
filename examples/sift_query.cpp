#include <assert.h>
#include <bits/stdc++.h>

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
using namespace std;
namespace hy {

struct SCC {
  using Graph = std::vector<std::vector<int>>;
  int n, scc_num = 0;
  const Graph &G;
  Graph S;
  std::vector<int> id, topo;
  std::vector<std::vector<int>> components;
  SCC(const Graph &G) : n((int)G.size()), G(G), id(n, -1) {
    std::vector<bool> vis(n);
    std::vector<int> post;
    for (int i = 0; i < n; ++i)
      if (!vis[i]) dfs1(G, i, vis, post);
    Graph rG(n);
    for (int i = 0; i < n; ++i)
      for (auto v : G[i]) rG[v].push_back(i);
    for (int i = n - 1; i >= 0; --i)
      if (id[post[i]] == -1) {
        components.push_back({});
        dfs2(rG, post[i], scc_num);
        scc_num++;
      }
    S.resize(scc_num);
    std::vector<int> deg(scc_num);
    for (int i = 0; i < n; ++i)
      for (auto v : G[i]) {
        if (int x = id[i], y = id[v]; x != y) S[x].push_back(y), deg[y]++;
      }
    std::queue<int> q;
    topo.reserve(scc_num);
    for (int i = 0; i < scc_num; ++i)
      if (!deg[i]) q.push(i);
    while (q.size()) {
      int u = q.front();
      q.pop();
      topo.push_back(u);
      for (auto v : S[u])
        if (--deg[v] == 0) q.push(v);
    }
  }

  void dfs1(const Graph &G, int u, std::vector<bool> &vis,
            std::vector<int> &post) {
    vis[u] = true;
    for (auto v : G[u]) {
      if (!vis[v]) {
        dfs1(G, v, vis, post);
      }
    }
    post.push_back(u);
  }

  void dfs2(const Graph &rG, int u, int idx) {
    id[u] = idx;
    components[idx].push_back(u);
    for (auto v : rG[u])
      if (id[v] == -1) dfs2(rG, v, idx);
  }
};

}  // namespace hy
constexpr int topk = 10, iters = 5;

void TEST() {
  int n, nq, dim, gt_k;
  float *base;
  float *query;
  int *gt;
  read_vecs<float>("../../tests/data/sift/sift_base.fvecs", base, n, dim);
  read_vecs<float>("../../tests/data/sift/sift_query.fvecs", query, nq, dim);
  read_vecs<int>("../../tests/data/sift/sift_groundtruth.ivecs", gt, nq, gt_k);

  auto space = new hnswlib::L2Space(dim);

  int cnt = 0;
  auto hnsw = new hnswlib::HierarchicalNSW<float>(space, "hnsw_M16_C100.index");
  vector<int> efs = {16, 24, 30, 36, 48, 64, 96, 128};
  for (auto ef : efs) {
    hy::OnlineEV<double> ev;
    hnsw->setEf(ef);
    for (int iter = 0; iter <= iters; ++iter) {
      int64_t cnt = 0;
      auto st = std::chrono::high_resolution_clock::now();
      // #pragma omp parallel for
      for (int i = 0; i < nq; ++i) {
        auto res = hnsw->searchKnn(query + i * dim, topk);
        std::unordered_set<int> st(gt + i * gt_k, gt + i * gt_k + topk);
        while (res.size()) {
          auto [dist, id] = res.top();
          res.pop();
          if (st.count(id)) {
            cnt++;
          }
        }
      }
      auto ed = std::chrono::high_resolution_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(ed - st)
              .count() /
          1000000.0;
      if (iter == 0) {
        double recall = double(cnt) / nq / topk;
        // writer.write((char *)&recall, sizeof(recall));
        std::cout << "ef: " << ef << "\n";
        std::cout << "  recall: " << recall << "\n";
      } else {
        auto qps = nq / elapsed;
        ev.insert(qps);
        // std::cout << ev.E() << std::endl;
      }
    }
    double qps = ev.E();
    std::cout << "  qps: " << qps << "\n";
    // writer.write((char *)&qps, sizeof(qps));
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
