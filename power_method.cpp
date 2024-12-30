#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

/*
 * Usage:
 *   ./power_method filename [--criterion 1|2] [--max_iter int] [--tol float]
 *
 * Arguments:
 *   filename    - path to the input mtx file
 *   --criterion - convergence criterion 1 or 2 (default: 1)
 *   --max_iter  - maximum number of iterations for the power method (default: 99)
 *   --tol       - tolerance for convergence (default: 1e-7)
 *
 * Example:
 *   ./power_method flickr.mtx --criterion 1 --max_iter 100 --tol 1e-6
 */

struct SparseMatrix {
    std::vector<float> values;
    std::vector<size_t> row_idx;
    std::vector<size_t> col_idx;
    const size_t rows;
    const size_t cols;
    const size_t nnz;

    SparseMatrix(size_t r, size_t c, size_t num_nonzero)
        : rows(r), cols(c), nnz(num_nonzero) {
        try {
            values.reserve(num_nonzero);
            row_idx.reserve(num_nonzero);
            col_idx.reserve(num_nonzero);
        } catch (const std::bad_alloc& e) {
            throw std::runtime_error("Failed to allocate sparse matrix arrays for " +
                                     std::to_string(num_nonzero) + " entries");
        }
    }

    float at(size_t i, size_t j) const {
        for (size_t k = 0; k < nnz; ++k) {
            if (row_idx[k] == i && col_idx[k] == j) {
                return values[k];
            }
        }
        throw std::runtime_error("No entry at position (" + std::to_string(i) + "," + std::to_string(j) + ")");
    }

    bool is_primitive() const {
        for (size_t k = 0; k < nnz; ++k) {
            if (values[k] <= 0) {
                return false;
            }
        }
        return true;
    }

    bool is_symmetric() const {
        for (size_t k = 0; k < nnz; ++k) {
            size_t i = row_idx[k];
            size_t j = col_idx[k];
            float value = values[k];
            bool found_symmetric = false;
            for (size_t l = 0; l < nnz; ++l) {
                if (row_idx[l] == j && col_idx[l] == i && values[l] == value) {
                    found_symmetric = true;
                    break;
                }
            }
            if (!found_symmetric) {
                return false;
            }
        }
        return true;
    }
};

SparseMatrix load_mtx_matrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::cout << "Loading matrix from file: " << filename << std::endl;
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty file");
    }
    do {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file while reading header");
        }
    } while (line[0] == '%');
    std::istringstream iss(line);
    size_t rows, cols, num_entries;
    if (!(iss >> rows >> cols >> num_entries)) {
        throw std::runtime_error("Failed to read matrix dimensions");
    }
    std::cout << "Creating " << rows << "x" << cols << " matrix with " << num_entries << " non-zero entries" << std::endl;
    SparseMatrix matrix(rows, cols, num_entries);
    size_t progress_step = num_entries / 20;
    if (progress_step == 0) {
        progress_step = 1;
    }
    std::cout << "Reading entries..." << std::endl;
    for (size_t i = 0; i < num_entries; ++i) {
        if (i % progress_step == 0) {
            std::cout << "\rProgress: " << (i * 100 / num_entries) << "%" << std::flush;
        }
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file at entry " + std::to_string(i));
        }
        iss.clear();
        iss.str(line);
        size_t row, col;
        if (!(iss >> row >> col)) {
            throw std::runtime_error("Failed to parse entry at line " + std::to_string(i));
        }
        matrix.values.push_back(1.0f);
        if (row == 0 || col == 0 || row > rows || col > cols) {
            throw std::runtime_error("Invalid indices at entry " + std::to_string(i) +
                                     ": (" + std::to_string(row) + "," + std::to_string(col) + ")");
        }
        matrix.row_idx.push_back(row - 1);
        matrix.col_idx.push_back(col - 1);
    }
    std::cout << "\rProgress: 100%" << std::endl;
    std::cout << "Matrix loaded successfully!" << std::endl;
    return matrix;
}

float l1_norm(const std::vector<float>& x) {
    float sum = 0.0f;
    for (const auto& val : x) {
        sum += std::abs(val);
    }
    return sum;
}

float l2_norm(const std::vector<float>& x) {
    float sum = 0.0f;
    for (const auto& val : x) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

float dot(const std::vector<float>& x, const std::vector<float>& y) {
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

std::vector<float> dot(SparseMatrix& A, const std::vector<float>& x) {
    std::vector<float> result(A.rows, 0.0f);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < A.cols; ++j) {
            result[i] += A.at(i, j) * x[j];
        }
    }
    return result;
}

std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

std::vector<float> multiply(const std::vector<float>& x, float scalar) {
    std::vector<float> result = x;
    for (auto& val : result) {
        val *= scalar;
    }
    return result;
}

std::vector<float> multiply(const SparseMatrix& A, const std::vector<float>& vec) {
    std::vector<float> result(A.rows, 0.0f);
    for (size_t i = 0; i < A.nnz; ++i) {
        result[A.row_idx[i]] += A.values[i] * vec[A.col_idx[i]];
    }
    return result;
}

void check_primitivness(const SparseMatrix& A) {
    auto start = std::chrono::high_resolution_clock::now();
    bool is_primitive = A.is_primitive();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;
    std::cout << "Matrix is " << (is_primitive ? "primitive" : "not primitive") << " (exec time: " << dur.count() << " milliseconds)" << std::endl;
}

void check_symmetry(const SparseMatrix& A) {
    auto start = std::chrono::high_resolution_clock::now();
    bool is_symmetric = A.is_symmetric();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;
    std::cout << "Matrix is " << (is_symmetric ? "symmetric" : "not symmetric") << " (exec time: " << dur.count() << " milliseconds)" << std::endl;
}

void check_initial_vector(const SparseMatrix& A, const std::vector<float>& x, float tol) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> Ax = multiply(A, x);
    bool is_zero = l2_norm(Ax) < tol;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;
    std::cout << "Matrix A * initial vector is a " << (is_zero ? "zero" : "non-zero") << " vector (exec time: " << dur.count() << " milliseconds)" << std::endl;
    if (is_zero) {
        throw std::runtime_error("Initial vector is a zero vector");
    }
}

std::tuple<float, std::vector<float>, int> power_method(const SparseMatrix& A, int cri, int max_iter, float tol) {
    std::vector<float> x_prev(A.rows, 1.0f / A.rows);
    check_initial_vector(A, x_prev, tol);

    for (int k = 1; k <= max_iter; ++k) {
        std::vector<float> Ax_prev = multiply(A, x_prev);
        std::vector<float> x_curr = multiply(Ax_prev, 1.0f / l1_norm(Ax_prev));
        std::vector<float> Ax_curr = multiply(A, x_curr);
        float lambda_1 = dot(x_curr, Ax_curr) / std::pow(l2_norm(x_curr), 2);
        if (cri == 1) {
            if (l1_norm(subtract(x_curr, x_prev)) < tol) {
                return {lambda_1, x_curr, k};
            }
        } else if (cri == 2) {
            std::vector<float> residual = subtract(Ax_curr, multiply(x_curr, lambda_1));
            if (l1_norm(residual) < tol) {
                return {lambda_1, x_curr, k};
            }
        }
        x_prev = x_curr;
        std::cout << "k: " << k << ", lambda_1: " << lambda_1 << std::endl;
    }
    return {0.0f, std::vector<float>(), max_iter};
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " filename [--criterion 1|2] [--max_iter int] [--tol float]" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    int criterion = 1;
    int max_iter = 99;
    float tol = 1e-7f;
    for (int i = 2; i < argc; i += 2) {
        std::string arg = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "Missing value for argument: " << arg << std::endl;
            return 1;
        }
        if (arg == "--criterion")
            criterion = std::stoi(argv[i + 1]);
        else if (arg == "--max_iter")
            max_iter = std::stoi(argv[i + 1]);
        else if (arg == "--tol")
            tol = std::stof(argv[i + 1]);
    }

    SparseMatrix A = load_mtx_matrix(filename);
    check_primitivness(A);
    check_symmetry(A);

    std::cout << "Criterion for convergence: " << criterion << std::endl;
    auto [eigenvalue, eigenvector, iterations] = power_method(A, criterion, max_iter, tol);
    if (!eigenvector.empty()) {
        float norm = l1_norm(eigenvector);
        std::vector<std::pair<int, float>> components;
        for (size_t i = 0; i < eigenvector.size(); ++i) {
            eigenvector[i] /= norm;
            components.push_back({i, eigenvector[i]});
        }
        std::sort(components.begin(), components.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cout << "5 largest components (index, value):" << std::endl;
        for (int i = 0; i < std::min(5, (int)components.size()); ++i) {
            std::cout << "(" << components[i].first << ", "
                      << std::fixed << std::setprecision(5) << components[i].second << ")" << std::endl;
        }
        std::cout << "Approximate value of lambda_1: " << std::fixed << std::setprecision(5) << eigenvalue << std::endl;
        std::cout << "Number of iterations: " << iterations << std::endl;
    } else {
        std::cout << "Convergence criterion was not met within the maximum number of iterations." << std::endl;
    }
    return 0;
}