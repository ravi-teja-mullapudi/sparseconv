#include <iostream>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>
#include <stdlib.h>

inline bool is_nearly_equal(float x, float y)
{
    const float epsilon = 1e-5;
    return std::abs(x - y) <= epsilon * std::abs(x);
}

template <typename F>
double benchmark(int samples, int iterations, F op) {
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < samples; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < iterations; j++) {
            op();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e6;
        if (dt < best) best = dt;
    }
    return best / iterations;
}

// Indexing function into a 4-d tensor
inline unsigned int I4(unsigned int i1, unsigned int i2, unsigned int i3, unsigned int i4,
                       unsigned int d1, unsigned int d2, unsigned int d3, unsigned int d4) {
    return i1 * d2 * d3 * d4 + i2 * d3 * d4 + i3 * d4 + i4;
}

// Convolution layer parameters
// b -- batch size
// h -- height input
// w -- width input
// c_in -- input channels
// c_out -- output channels
// f -- filter size
// s -- stride
// p -- padding
//
// For more information http://cs231n.github.io/convolutional-networks/
// A guide to convolution arithmetic for deep learning https://arxiv.org/pdf/1603.07285.pdf

// F_in is a tensor of dimension b x (h+2p) x (w+2p) x c_in
// F_out is a tensor of dimension b x ((h - f + 2p)/s + 1) x ((w - f + 2p)/s + 1) x c_out
// W is a tensor of dimension c_out x c_in x f x f
// M is a tensor of dimension b x ((h - f + 2p)/s + 1) x ((w - f + 2p)/s + 1)
int conv_naive(float *F_in, float *W, float *F_out, bool *M,
               unsigned int b, unsigned int h, unsigned int w,
               unsigned int c_in, unsigned int c_out,
               unsigned int f, unsigned int s, unsigned int p) {

    unsigned int h_out = (h - f + 2*p)/s + 1;
    unsigned int w_out = (w - f + 2*p)/s + 1;
    for (unsigned int b_i = 0; b_i < b; b_i++) {
#pragma omp parallel for
        for (unsigned int h_i = 0; h_i < h_out; h_i++) {
            for (unsigned int w_i = 0; w_i < w_out; w_i++) {
                // Skip computation where the mask is zero
                if (M != nullptr) {
                    unsigned int M_offset = I4(0, b_i, h_i, w_i, 1, b, h_out, w_out);
                    if (!M[M_offset]) {
                        continue;
                    }
                }

//#pragma vector aligned
#pragma omp simd
                for (unsigned int c_out_i = 0; c_out_i < c_out; c_out_i++) {
                    unsigned int out_offset = I4(b_i, h_i, w_i, c_out_i,
                                                 b, h_out, w_out, c_out);
                    //assert(out_offset >= 0 && out_offset < (b * h_out * w_out * c_out));
                    F_out[out_offset] = 0.0f;
                    for (unsigned int c_in_i = 0; c_in_i < c_in; c_in_i++) {
                        unsigned int in_offset = I4(b_i, s*h_i + p, s*w_i + p, c_in_i,
                                                    b, h, w, c_in);
                        //assert(in_offset >= 0 && in_offset < (b * h * w * c_in));
                        for (unsigned int f_h = 0; f_h < f; f_h++) {
                            for (unsigned int f_w = 0; f_w < f; f_w++) {
                                unsigned int w_offset = I4(c_out_i, c_in_i, f_h, f_w,
                                                           c_out, c_in, f, f);
                                //assert(w_offset >= 0 && w_offset < (c_out * c_in * f * f));
                                F_out[out_offset] += W[w_offset] * F_in[in_offset];
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int conv_tiled_spatial(float *F_in, float *W, float *F_out, bool *M,
                       unsigned int b, unsigned int h, unsigned int w,
                       unsigned int c_in, unsigned int c_out,
                       unsigned int f, unsigned int s, unsigned int p,
                       unsigned int tile_size) {

    unsigned int h_out = (h - f + 2*p)/s + 1;
    unsigned int w_out = (w - f + 2*p)/s + 1;

    unsigned int h_tiles = h_out/tile_size + (h_out%tile_size != 0);
    unsigned int w_tiles = w_out/tile_size + (w_out%tile_size != 0);

    for (unsigned int b_i = 0; b_i < b; b_i++) {
#pragma omp parallel for collapse(2)
        for (unsigned int h_t = 0; h_t < h_tiles; h_t++) {
            for (unsigned int w_t = 0; w_t < w_tiles; w_t++) {
                unsigned int h_start = h_t * tile_size;
                unsigned int h_end = std::min((h_t + 1)*tile_size, h_out);
                unsigned int w_start = w_t * tile_size;
                unsigned int w_end = std::min((w_t + 1)*tile_size, w_out);
                for (unsigned int h_i = h_start; h_i < h_end; h_i++) {
                    for (unsigned int w_i = w_start; w_i < w_end; w_i++) {
                        // Skip computation where the mask is zero
                        if (M != nullptr) {
                            unsigned int M_offset = I4(0, b_i, h_i, w_i, 1, b, h_out, w_out);
                            if (!M[M_offset]) {
                                continue;
                            }
                        }

//#pragma vector aligned
#pragma omp simd
                        for (unsigned int c_out_i = 0; c_out_i < c_out; c_out_i++) {
                            unsigned int out_offset = I4(b_i, h_i, w_i, c_out_i,
                                    b, h_out, w_out, c_out);
                            //assert(out_offset >= 0 && out_offset < (b * h_out * w_out * c_out));
                            F_out[out_offset] = 0.0f;
                            for (unsigned int c_in_i = 0; c_in_i < c_in; c_in_i++) {
                                unsigned int in_offset = I4(b_i, s*h_i + p, s*w_i + p, c_in_i,
                                        b, h, w, c_in);
                                //assert(in_offset >= 0 && in_offset < (b * h * w * c_in));
                                for (unsigned int f_h = 0; f_h < f; f_h++) {
                                    for (unsigned int f_w = 0; f_w < f; f_w++) {
                                        unsigned int w_offset = I4(c_out_i, c_in_i, f_h, f_w,
                                                c_out, c_in, f, f);
                                        //assert(w_offset >= 0 && w_offset < (c_out * c_in * f * f));
                                        F_out[out_offset] += W[w_offset] * F_in[in_offset];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

// Randomly initialize a 4d-tensor
int generate_random_tensor(float *T, unsigned int d1, unsigned int d2, unsigned int d3, unsigned int d4) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (unsigned int i1 = 0; i1 < d1; i1++) {
        for (unsigned int i2 = 0; i2 < d2; i2++) {
            for (unsigned int i3 = 0; i3 < d3; i3++) {
                for (unsigned int i4 = 0; i4 < d4; i4++) {
                    unsigned int offset = I4(i1, i2, i3, i4, d1, d2, d3, d4);
                    T[offset] =  distribution(generator);
                }
            }
        }
    }
    return 0;
}

unsigned int generate_sparsity_pattern(bool *T,  unsigned int d1, unsigned int d2, unsigned int d3,
                                       float sparsity) {

    assert(sparsity >= 0 && sparsity <= 1);
    std::default_random_engine generator;
    std::bernoulli_distribution distribution(sparsity);
    unsigned int nnz = 0;
    for (unsigned int i1 = 0; i1 < d1; i1++) {
        for (unsigned int i2 = 0; i2 < d2; i2++) {
            for (unsigned int i3 = 0; i3 < d3; i3++) {
                unsigned int offset = I4(0, i1, i2, i3, 1, d1, d2, d3);
                T[offset] =  distribution(generator);
                if (T[offset]) {
                    nnz++;
                }
            }
        }
    }
    return nnz;
}

int main() {
    unsigned int f = 3;
    unsigned int c_in = 16;
    unsigned int c_out = 64;
    unsigned int h = 256;
    unsigned int w = 256;
    unsigned int p = 1;
    unsigned int b = 1;
    unsigned int s = 1;
    unsigned int h_out = ((h - f + 2*p)/s + 1);
    unsigned int w_out = ((w - f + 2*p)/s + 1);

    float *F_in = (float*) aligned_alloc(64, sizeof(float) * b * (h + 2*p) * (w + 2*p) * c_in);
    float *W = (float*) aligned_alloc(64, sizeof(float) * c_out * c_in * f * f);
    float *F_out = (float*) aligned_alloc(64, sizeof(float) * b * h_out * w_out * c_out);
    bool *M = (bool*) aligned_alloc(64, sizeof(bool) *b * h_out * w_out);

    generate_random_tensor(F_in, b, h + 2*p, w + 2*p, c_in);
    generate_random_tensor(W, c_out, c_in, f, f);
    generate_random_tensor(F_out, b, h_out, w_out, c_out);
    unsigned int nnz = generate_sparsity_pattern(M, b, h_out, w_out, 0.1);

    float time_dense = benchmark(5, 1, [&]() {
        conv_naive(F_in, W, F_out, nullptr, b, h, w, c_in, c_out, f, s, p);
    });

    float gfops_dense = ((float)b * c_in * h_out * w_out * c_out * f * f)/(1e09);
    float gflops_dense = (gfops_dense)/time_dense;
    std::cout << "Dense Time : " << 1000 * time_dense << "ms " << std::endl;
    std::cout << "GFLOPS : " << gflops_dense << std::endl;

    unsigned int tile_size = 8;
    float time_tiled_spatial = benchmark(5, 1, [&]() {
        conv_tiled_spatial(F_in, W, F_out, nullptr, b, h, w, c_in, c_out, f, s, p, tile_size);
    });

    float gflops_tiled_spatial = (gfops_dense)/time_tiled_spatial;
    std::cout << "Dense Time : " << 1000 * time_tiled_spatial << "ms " << std::endl;
    std::cout << "GFLOPS : " << gflops_tiled_spatial << std::endl;

    float time_sparse = benchmark(5, 1, [&]() {
        conv_naive(F_in, W, F_out, M, b, h, w, c_in, c_out, f, s, p);
    });

    float gfops_sparse = ((float)b * c_in * nnz * c_out * f * f)/(1e09);
    float gflops_sparse = (gfops_sparse)/time_sparse;
    std::cout << "Sparse Time: " << 1000 * time_sparse << "ms " << std::endl;
    std::cout << "GFLOPS : " << gflops_sparse << std::endl;

    free(F_in);
    free(W);
    free(F_out);
    free(M);

    return 0;
}
