#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numeric>  // For std::iota

namespace py = pybind11;

class Sequence {
public:
    int seq;
    int id;

    Sequence(int seq, int id = 0) : seq(seq), id(id) {}

    bool operator<(const Sequence& other) const {
        return seq < other.seq;
    }

    std::string to_string() const {
        return std::to_string(id) + "-" + std::to_string(seq);
    }
};

void print_seqs(const std::vector<Sequence>& seqs) {
    if (seqs.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }
    std::cout << "[";
    for (size_t i = 0; i < seqs.size() - 1; ++i) {
        std::cout << seqs[i].to_string() << ", ";
    }
    std::cout << seqs.back().to_string() << "]" << std::endl;
}

std::vector<int> get_lens(const std::vector<Sequence>& seqs) {
    std::vector<int> lens;
    for (const auto& seq : seqs) {
        lens.push_back(seq.seq);
    }
    return lens;
}

class SeqBucket {
public:
    int boundary;
    std::vector<Sequence> seqs;
    int size;

    SeqBucket(int boundary) : boundary(boundary), size(0) {}

    void add_seqs(const std::vector<Sequence>& seqs) {
        size += seqs.size();
        this->seqs.insert(this->seqs.end(), seqs.begin(), seqs.end());
    }

    std::vector<Sequence> random_pop_seqs(int num = 1) {
        if (size == 0) {
            return {};  
        }
        if (num > size) {
            throw std::invalid_argument("Num cannot be greater than the bucket size.");
        }

        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);  
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        indices.resize(num);
        
        std::sort(indices.begin(), indices.end(), std::greater<int>());

        std::vector<Sequence> poped_seqs;
        for (int index : indices) {
            poped_seqs.push_back(seqs[index]);  
            seqs.erase(seqs.begin() + index);  
        }

        size -= num;
        return poped_seqs;
    }

    void print() const {
        std::cout << "[Bucket] Boundary: " << boundary << ", Size: " << size << ", Seqs: ";
        print_seqs(seqs);
    }
};

std::pair<std::vector<SeqBucket>, double> bucketing_seqs(std::vector<Sequence>& sequences, int B) {
    int n = sequences.size();
    std::vector<Sequence> sequences_sorted = sequences;
    std::sort(sequences_sorted.begin(), sequences_sorted.end());

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> dp(n + 1, std::vector<double>(B + 1, inf));
    dp[0][0] = 0;

    std::vector<int> prefix_sum(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        prefix_sum[i] = prefix_sum[i - 1] + sequences_sorted[i - 1].seq;
    }

    for (int b = 1; b <= B; ++b) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < i; ++j) {
                double cost = sequences_sorted[i - 1].seq * (i - j) - (prefix_sum[i] - prefix_sum[j]);
                if (dp[j][b - 1] + cost < dp[i][b]) {
                    dp[i][b] = dp[j][b - 1] + cost;
                }
            }
        }
    }

    double avg_error = dp[n][B] / n;

    std::vector<SeqBucket> buckets;
    int current = n;
    double last_boundary = inf;
    for (int b = B; b > 0; --b) {
        for (int i = 0; i < current; ++i) {
            double cost = sequences_sorted[current - 1].seq * (current - i) - (prefix_sum[current] - prefix_sum[i]);
            if (std::abs(dp[i][b - 1] + cost - dp[current][b]) < 1e-9) {
                int boundary = sequences_sorted[current - 1].seq;
                SeqBucket bucket(boundary);
                bucket.add_seqs(std::vector<Sequence>(sequences_sorted.begin() + i, sequences_sorted.begin() + current));
                buckets.push_back(bucket);
                current = i;
                break;
            }
        }
    }

    if (current > 0) {
        SeqBucket bucket(last_boundary);
        bucket.add_seqs(std::vector<Sequence>(sequences_sorted.begin(), sequences_sorted.begin() + current));
        buckets.push_back(bucket);
    }

    std::reverse(buckets.begin(), buckets.end());
    return {buckets, avg_error};
}


std::vector<std::vector<Sequence>> chunk_globalbatch_sort_consec(std::vector<Sequence>& seqs_gb, int mb_num) {
    std::sort(seqs_gb.begin(), seqs_gb.end(), [](const Sequence& a, const Sequence& b) {
        return a.seq > b.seq;
    });

    int n = seqs_gb.size();
    std::vector<int> prefix_sum(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        prefix_sum[i] = prefix_sum[i - 1] + seqs_gb[i - 1].seq;
    }

    const int inf = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(mb_num + 1, inf));
    std::vector<std::vector<int>> partition(n + 1, std::vector<int>(mb_num + 1, 0));

    dp[0][0] = 0;

    for (int k = 1; k <= mb_num; ++k) {
        for (int i = 1; i <= n; ++i) {
            for (int j = k - 1; j < i; ++j) {
                int cost = prefix_sum[i] - prefix_sum[j];
                if (std::max(dp[j][k - 1], cost) < dp[i][k]) {
                    dp[i][k] = std::max(dp[j][k - 1], cost);
                    partition[i][k] = j;
                }
            }
        }
    }

    std::vector<std::vector<Sequence>> micro_batches;
    int k = mb_num;
    int index = n;
    while (k > 0) {
        int start_index = partition[index][k];
        micro_batches.push_back(std::vector<Sequence>(seqs_gb.begin() + start_index, seqs_gb.begin() + index));
        index = start_index;
        k--;
    }

    std::reverse(micro_batches.begin(), micro_batches.end());
    return micro_batches;
}

std::vector<std::vector<Sequence>> chunk_globalbatch_distribution(std::vector<Sequence>& seqs_gb, int mb_num) {
    int n = seqs_gb.size();
    std::vector<int> prefix_sum(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        prefix_sum[i] = prefix_sum[i - 1] + seqs_gb[i - 1].seq;
    }

    const int inf = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(mb_num + 1, inf));
    std::vector<std::vector<int>> partition(n + 1, std::vector<int>(mb_num + 1, 0));

    dp[0][0] = 0;  

    for (int k = 1; k <= mb_num; ++k) {
        for (int i = 1; i <= n; ++i) {
            for (int j = k - 1; j < i; ++j) {
                int cost = prefix_sum[i] - prefix_sum[j];
                if (std::max(dp[j][k - 1], cost) < dp[i][k]) {
                    dp[i][k] = std::max(dp[j][k - 1], cost);
                    partition[i][k] = j;
                }
            }
        }
    }

    std::vector<std::vector<Sequence>> micro_batches;
    int k = mb_num;
    int index = n;
    while (k > 0) {
        int start_index = partition[index][k];
        micro_batches.push_back(std::vector<Sequence>(seqs_gb.begin() + start_index, seqs_gb.begin() + index));
        index = start_index;
        k--;
    }

    std::reverse(micro_batches.begin(), micro_batches.end());

    return micro_batches;
}

std::vector<std::vector<Sequence>> chunk_globalbatch(std::vector<Sequence>& seqs_gb, int mb_num, const std::string& chunk_alg = "sort_consec") {
    if (chunk_alg == "sort_consec") {
        return chunk_globalbatch_sort_consec(seqs_gb, mb_num);
    } else if (chunk_alg == "distribution") {
        return chunk_globalbatch_distribution(seqs_gb, mb_num);
    } else {
        throw std::invalid_argument("Invalid chunk algorithm");
    }
}

// Pybind11 bindings
PYBIND11_MODULE(sequence_module, m) {
    py::class_<Sequence>(m, "Sequence")
        .def(py::init<int, int>(), py::arg("seq"), py::arg("id") = 0)
        .def("__lt__", &Sequence::operator<)
        .def("__str__", &Sequence::to_string)
        .def_readwrite("seq", &Sequence::seq)  // Expose seq for read/write access
        .def_readwrite("id", &Sequence::id);   // Expose id for read/write access

    m.def("print_seqs", &print_seqs);
    m.def("get_lens", &get_lens);

    py::class_<SeqBucket>(m, "SeqBucket")
        .def(py::init<int>(), py::arg("boundary"))
        .def("add_seqs", &SeqBucket::add_seqs)
        .def("random_pop_seqs", &SeqBucket::random_pop_seqs)
        .def("print", &SeqBucket::print)
        .def_readwrite("boundary", &SeqBucket::boundary)  // Expose boundary for read/write access
        .def_readwrite("seqs", &SeqBucket::seqs)   // Expose seqs for read/write access
        .def_readwrite("size", &SeqBucket::size);   // Expose seqs for read/write access

    m.def("bucketing_seqs", &bucketing_seqs);
    m.def("chunk_globalbatch", &chunk_globalbatch);
}
