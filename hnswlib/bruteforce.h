#pragma once
#include <assert.h>

#include <algorithm>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace hnswlib {
template <typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 public:
  char *data_;
  size_t maxelements_;
  size_t cur_element_count;
  size_t size_per_element_;

  size_t data_size_;
  DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_;
  std::mutex index_lock;

  std::unordered_map<labeltype, size_t> dict_external_to_internal;

  BruteforceSearch(SpaceInterface<dist_t> *s)
      : data_(nullptr),
        maxelements_(0),
        cur_element_count(0),
        size_per_element_(0),
        data_size_(0),
        dist_func_param_(nullptr) {}

  BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
      : data_(nullptr),
        maxelements_(0),
        cur_element_count(0),
        size_per_element_(0),
        data_size_(0),
        dist_func_param_(nullptr) {
    loadIndex(location, s);
  }

  BruteforceSearch(SpaceInterface<dist_t> *s, size_t maxElements) {
    maxelements_ = maxElements;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    size_per_element_ = data_size_ + sizeof(labeltype);
    data_ = (char *)malloc(maxElements * size_per_element_);
    cur_element_count = 0;
  }

  ~BruteforceSearch() { free(data_); }

  void addPoint(const void *datapoint, labeltype label,
                bool replace_deleted = false) {
    int idx;
    {
      std::unique_lock<std::mutex> lock(index_lock);

      auto search = dict_external_to_internal.find(label);
      if (search != dict_external_to_internal.end()) {
        idx = search->second;
      } else {
        idx = cur_element_count;
        dict_external_to_internal[label] = idx;
        cur_element_count++;
      }
    }
    memcpy(data_ + size_per_element_ * idx + data_size_, &label,
           sizeof(labeltype));
    memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
  }

  void removePoint(labeltype cur_external) {
    size_t cur_c = dict_external_to_internal[cur_external];

    dict_external_to_internal.erase(cur_external);

    labeltype label =
        *((labeltype *)(data_ + size_per_element_ * (cur_element_count - 1) +
                        data_size_));
    dict_external_to_internal[label] = cur_c;
    memcpy(data_ + size_per_element_ * cur_c,
           data_ + size_per_element_ * (cur_element_count - 1),
           data_size_ + sizeof(labeltype));
    cur_element_count--;
  }

  void searchKnn(const void *query_data, size_t k, int *) const {}

  void saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, maxelements_);
    writeBinaryPOD(output, size_per_element_);
    writeBinaryPOD(output, cur_element_count);

    output.write(data_, maxelements_ * size_per_element_);

    output.close();
  }

  void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
    std::ifstream input(location, std::ios::binary);
    std::streampos position;

    readBinaryPOD(input, maxelements_);
    readBinaryPOD(input, size_per_element_);
    readBinaryPOD(input, cur_element_count);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    size_per_element_ = data_size_ + sizeof(labeltype);
    data_ = (char *)malloc(maxelements_ * size_per_element_);

    input.read(data_, maxelements_ * size_per_element_);

    input.close();
  }
};
}  // namespace hnswlib
