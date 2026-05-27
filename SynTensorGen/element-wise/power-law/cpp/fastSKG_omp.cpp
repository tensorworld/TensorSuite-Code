#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

struct Tensor {
    int num_modes = 0;
    vector<long> mode_size;
    vector<double> values;
    long nnz = 0;
};

static inline bool is_power_of_two(long long x) {
    return x > 0 && (x & (x - 1)) == 0;
}

static inline long long ipow_ll(long long base, long exp) {
    long long res = 1;
    for (long i = 0; i < exp; i++) res *= base;
    return res;
}

static inline void append_int(string& s, long v) {
    char buf[32];
    auto [ptr, ec] = to_chars(buf, buf + sizeof(buf), v);
    s.append(buf, ptr);
}

static bool read_tensor_suite_seed(const string& filepath, Tensor& T, int index_base) {
    ifstream fin(filepath);
    if (!fin.is_open()) {
        cerr << "Error: cannot open seed " << filepath << endl;
        return false;
    }
    string line;
    if (!getline(fin, line) || line != "%%TensorSuite-TNS") {
        cerr << "Error: seed must use TensorSuite-TNS magic" << endl;
        return false;
    }
    if (!getline(fin, line) || line.rfind("% version:", 0) != 0) return false;
    if (!getline(fin, line) || line.rfind("% name:", 0) != 0) return false;
    if (!getline(fin, line)) return false;
    stringstream header(line);
    header >> T.num_modes;
    if (T.num_modes <= 0) return false;
    T.mode_size.assign(T.num_modes, 0);
    for (int d = 0; d < T.num_modes; d++) header >> T.mode_size[d];
    long expected_nnz = 0;
    header >> expected_nnz;
    long total_elems = 1;
    for (long s : T.mode_size) {
        if (s <= 0) return false;
        total_elems *= s;
    }
    T.values.assign(total_elems, 0.0);
    T.nnz = 0;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<long> idx(T.num_modes);
        for (int d = 0; d < T.num_modes; d++) {
            ss >> idx[d];
            idx[d] -= index_base;
            if (idx[d] < 0 || idx[d] >= T.mode_size[d]) {
                cerr << "Error: seed coordinate out of range" << endl;
                return false;
            }
        }
        double v = 0.0;
        ss >> v;
        long flat = 0, mul = 1;
        for (int d = T.num_modes - 1; d >= 0; d--) {
            flat += idx[d] * mul;
            mul *= T.mode_size[d];
        }
        T.values[flat] = v;
        T.nnz++;
    }
    if (T.nnz != expected_nnz) {
        cerr << "Error: seed nnz mismatch" << endl;
        return false;
    }
    return true;
}

static int sampling(const double* P, const long long N) {
    static thread_local mt19937_64 gen(random_device{}());
    uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(gen);
    double acc = 0.0;
    for (long long i = 0; i < N; ++i) {
        acc += P[i];
        if (u <= acc) return (int)i;
    }
    return (int)(N - 1);
}

static bool fastSKGOmp(const Tensor& initiator,
                       long num_iter,
                       const string& outPath,
                       long user_assigned_nnz,
                       int index_base,
                       const string& name) {
    ofstream fout(outPath, ios::out | ios::trunc);
    if (!fout.is_open()) return false;
    const int D = initiator.num_modes;
    vector<long long> out_dims(D);
    for (int d = 0; d < D; d++) out_dims[d] = ipow_ll((long long)initiator.mode_size[d], num_iter);
    long long seedElementNum = 1;
    for (int i = 0; i < D; i++) seedElementNum *= initiator.mode_size[i];
    double E1 = 0.0;
    for (long long i = 0; i < seedElementNum; i++) E1 += initiator.values[i];
    long long E = user_assigned_nnz > 0 ? user_assigned_nnz : (long long)llround(pow(E1, num_iter));
    fout << "%%TensorSuite-TNS\n";
    fout << "% version: 0.1\n";
    fout << "% name: " << name << "\n";
    fout << D;
    for (int d = 0; d < D; d++) fout << " " << out_dims[d];
    fout << " " << E << "\n";
    vector<double> P(seedElementNum);
    for (long long i = 0; i < seedElementNum; i++) P[i] = initiator.values[i] / E1;
    vector<bool> is_pow2(D);
    vector<int> shift(D);
    for (int d = 0; d < D; d++) {
        long long ms = initiator.mode_size[d];
        if (is_power_of_two(ms)) {
            is_pow2[d] = true;
            shift[d] = __builtin_ctzll(ms);
        } else {
            is_pow2[d] = false;
        }
    }
    #pragma omp parallel
    {
        vector<long> x(D);
        vector<int> a(D);
        string out;
        out.reserve(1 << 20);
        const size_t FLUSH_TH = 1 << 20;
        #pragma omp for
        for (long long e = 0; e < E; e++) {
            fill(x.begin(), x.end(), 0L);
            for (int t = 0; t < num_iter; t++) {
                int seedIdx = sampling(P.data(), seedElementNum);
                long long q = seedIdx;
                for (int d = D - 1; d >= 0; --d) {
                    if (is_pow2[d]) {
                        a[d] = q & ((1LL << shift[d]) - 1);
                        q >>= shift[d];
                    } else {
                        a[d] = q % initiator.mode_size[d];
                        q /= initiator.mode_size[d];
                    }
                }
                for (int d = 0; d < D; d++) {
                    x[d] = x[d] * initiator.mode_size[d] + a[d];
                }
            }
            for (int d = 0; d < D; d++) {
                append_int(out, x[d] + index_base);
                out.push_back(' ');
            }
            out += "1\n";
            if (out.size() >= FLUSH_TH) {
                #pragma omp critical
                fout << out;
                out.clear();
            }
        }
        if (!out.empty()) {
            #pragma omp critical
            fout << out;
        }
    }
    return true;
}

struct Args {
    string inPath;
    string outPath;
    string name = "kronweave_raw";
    long num_iter = -1;
    long user_assigned_nnz = -1;
    int index_base = 0;
};

static bool parseCmdArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) args.inPath = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) args.outPath = argv[++i];
        else if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) args.num_iter = stol(argv[++i]);
        else if (strcmp(argv[i], "--nnz") == 0 && i + 1 < argc) args.user_assigned_nnz = stol(argv[++i]);
        else if (strcmp(argv[i], "--index-base") == 0 && i + 1 < argc) args.index_base = stoi(argv[++i]);
        else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) args.name = argv[++i];
        else return false;
    }
    return !(args.inPath.empty() || args.outPath.empty() || args.num_iter < 0) &&
           (args.index_base == 0 || args.index_base == 1);
}

int main(int argc, char** argv) {
    Args args;
    if (!parseCmdArgs(argc, argv, args)) {
        cerr << "Usage: " << argv[0] << " --seed <seed.tns> --out <out.tns> --iter <k> [--nnz N] [--index-base 0|1] [--name name]\n";
        return 1;
    }
    Tensor seed;
    if (!read_tensor_suite_seed(args.inPath, seed, args.index_base)) return 2;
    bool ok = fastSKGOmp(seed, args.num_iter, args.outPath, args.user_assigned_nnz, args.index_base, args.name);
    return ok ? 0 : 3;
}
