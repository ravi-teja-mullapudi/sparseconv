#include <iostream>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>
#include <stdlib.h>
#include <omp.h>

extern "C" {
#define restrict __restrict__
#include "third_party/FALCON/include/falcon.h"
#undef restrict
}

#define SF 2
#define SCALE(x) ((x)*SF)

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
// W is a tensor of dimension c_out x f x f x c_in
// M is a tensor of dimension b x ((h - f + 2p)/s + 1) x ((w - f + 2p)/s + 1)

template<unsigned int b, unsigned int h, unsigned int w,
         unsigned int c_in, unsigned int c_out, unsigned int f,
         unsigned int s, unsigned int p>
int conv_naive(float *F_in, float *W, float *F_out, bool *M) {
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
                        for (unsigned int f_h = 0; f_h < f; f_h++) {
                            for (unsigned int f_w = 0; f_w < f; f_w++) {
                                unsigned int in_offset = I4(b_i, s*h_i + f_h, s*w_i + f_w, c_in_i,
                                                            b, h, w, c_in);
                                //assert(in_offset >= 0 && in_offset < (b * (h+2*p) * (w+2*p) * c_in));
                                //unsigned int w_offset = I4(c_out_i, c_in_i, f_h, f_w,
                                //                           c_out, c_in, f, f);
                                unsigned int w_offset = I4(c_in_i, f_h, f_w, c_out_i,
                                                           c_in, f, f, c_out);
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

template<unsigned int b, unsigned int h, unsigned int w,
         unsigned int c_in, unsigned int c_out, unsigned int f,
         unsigned int s, unsigned int p>
int conv_naive_tmpl(float *F_in, float *W, float *F_out, bool *M) {

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

#pragma vector aligned
#pragma omp simd
                for (unsigned int c_out_i = 0; c_out_i < c_out; c_out_i++) {
                    unsigned int out_offset = I4(b_i, h_i, w_i, c_out_i,
                                                 b, h_out, w_out, c_out);
                    //assert(out_offset >= 0 && out_offset < (b * h_out * w_out * c_out));
                    F_out[out_offset] = 0.0f;
                    for (unsigned int c_in_i = 0; c_in_i < c_in; c_in_i++) {
                        for (unsigned int f_h = 0; f_h < f; f_h++) {
                            for (unsigned int f_w = 0; f_w < f; f_w++) {
                                unsigned int in_offset = I4(b_i, s*h_i + f_h, s*w_i + f_w, c_in_i,
                                                            b, h, w, c_in);
                                //assert(in_offset >= 0 && in_offset < (b * (h+2*p) * (w+2*p) * c_in));
                                //unsigned int w_offset = I4(c_out_i, c_in_i, f_h, f_w,
                                //                           c_out, c_in, f, f);
                                unsigned int w_offset = I4(c_in_i, f_h, f_w, c_out_i,
                                                           c_in, f, f, c_out);
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

template<unsigned int b, unsigned int h, unsigned int w,
         unsigned int c_in, unsigned int c_out, unsigned int f,
         unsigned int s, unsigned int p, unsigned int tile_h, unsigned int tile_w>
int conv_tiled_spatial(float *F_in, float *W, float *F_out, bool *M) {

    unsigned int h_out = (h - f + 2*p)/s + 1;
    unsigned int w_out = (w - f + 2*p)/s + 1;

    unsigned int h_tiles = h_out/tile_h + (h_out%tile_h != 0);
    unsigned int w_tiles = w_out/tile_w + (w_out%tile_w != 0);

    for (unsigned int b_i = 0; b_i < b; b_i++) {
#pragma omp parallel for collapse(2)
        for (unsigned int h_t = 0; h_t < h_tiles; h_t++) {
            for (unsigned int w_t = 0; w_t < w_tiles; w_t++) {
                unsigned int h_start = h_t * tile_h;
                unsigned int h_end = std::min((h_t + 1)*tile_h, h_out);
                unsigned int w_start = w_t * tile_w;
                unsigned int w_end = std::min((w_t + 1)*tile_w, w_out);
                for (unsigned int h_i = h_start; h_i < h_end; h_i++) {
                    for (unsigned int w_i = w_start; w_i < w_end; w_i++) {
                        // Skip computation where the mask is zero
                        if (M != nullptr) {
                            unsigned int M_offset = I4(0, b_i, h_i, w_i, 1, b, h_out, w_out);
                            if (!M[M_offset]) {
                                continue;
                            }
                        }

#pragma vector aligned
#pragma omp simd
                        for (unsigned int c_out_i = 0; c_out_i < c_out; c_out_i++) {
                            unsigned int out_offset = I4(b_i, h_i, w_i, c_out_i,
                                    b, h_out, w_out, c_out);
                            //assert(out_offset >= 0 && out_offset < (b * h_out * w_out * c_out));
                            F_out[out_offset] = 0.0f;
                            for (unsigned int c_in_i = 0; c_in_i < c_in; c_in_i++) {
                                for (unsigned int f_h = 0; f_h < f; f_h++) {
                                    for (unsigned int f_w = 0; f_w < f; f_w++) {
                                        unsigned int in_offset = I4(b_i, s*h_i + f_h, s*w_i + f_w, c_in_i,
                                                                    b, h, w, c_in);
                                        //assert(in_offset >= 0 && in_offset < (b * (h+2*p) * (w+2*p) * c_in));
                                        //unsigned int w_offset = I4(c_out_i, c_in_i, f_h, f_w,
                                        //                           c_out, c_in, f, f);
                                        unsigned int w_offset = I4(c_in_i, f_h, f_w, c_out_i,
                                                                   c_in, f, f, c_out);

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

template<unsigned int b, unsigned int h, unsigned int w,
         unsigned int c_in, unsigned int c_out, unsigned int f,
         unsigned int s, unsigned int p, unsigned int t1, unsigned int t2>
int conv_tiled_3d(float *F_in, float *W, float *F_out, bool *M) {

    unsigned int h_out = (h - f + 2*p)/s + 1;
    unsigned int w_out = (w - f + 2*p)/s + 1;

    unsigned int h_tiles = h_out/t1 + (h_out%t1 != 0);
    unsigned int w_tiles = w_out/t1 + (w_out%t1 != 0);
    unsigned int c_tiles = c_out/t2 + (c_out%t2 != 0);

    for (unsigned int b_i = 0; b_i < b; b_i++) {
#pragma omp parallel for collapse(2)
        for (unsigned int h_t = 0; h_t < h_tiles; h_t++) {
            for (unsigned int w_t = 0; w_t < w_tiles; w_t++) {
                for (unsigned int c_t = 0; c_t < c_tiles; c_t++) {
                    unsigned int c_start = c_t * t2;
                    unsigned int c_end = std::min((c_t + 1)*t2, c_out);
                    unsigned int h_start = h_t * t1;
                    unsigned int h_end = std::min((h_t + 1)*t1, h_out);
                    unsigned int w_start = w_t * t1;
                    unsigned int w_end = std::min((w_t + 1)*t1, w_out);
                    for (unsigned int h_i = h_start; h_i < h_end; h_i++) {
                        for (unsigned int w_i = w_start; w_i < w_end; w_i++) {
                            // Skip computation where the mask is zero
                            if (M != nullptr) {
                                unsigned int M_offset = I4(0, b_i, h_i, w_i, 1, b, h_out, w_out);
                                if (!M[M_offset]) {
                                    continue;
                                }
                            }

#pragma vector aligned
#pragma omp simd
                            for (unsigned int c_out_i = c_start; c_out_i < c_end; c_out_i++) {
                                unsigned int out_offset = I4(b_i, h_i, w_i, c_out_i,
                                                             b, h_out, w_out, c_out);
                                //assert(out_offset >= 0 && out_offset < (b * h_out * w_out * c_out));
                                F_out[out_offset] = 0.0f;
                                for (unsigned int c_in_i = 0; c_in_i < c_in; c_in_i++) {
                                    for (unsigned int f_h = 0; f_h < f; f_h++) {
                                        for (unsigned int f_w = 0; f_w < f; f_w++) {
                                            unsigned int in_offset = I4(b_i, s*h_i + f_h, s*w_i + f_w, c_in_i,
                                                                        b, h, w, c_in);
                                            //assert(in_offset >= 0 && in_offset < (b * (h+2*p) * (w+2*p) * c_in));
                                            //unsigned int w_offset = I4(c_out_i, c_in_i, f_h, f_w,
                                            //                           c_out, c_in, f, f);
                                            unsigned int w_offset = I4(c_in_i, f_h, f_w, c_out_i,
                                                                       c_in, f, f, c_out);
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
    }
    return 0;
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
// W is a tensor of dimension c_out x f x f x c_in
// M is a tensor of dimension b x ((h - f + 2p)/s + 1) x ((w - f + 2p)/s + 1)

int conv_winograd(float *F_in, float *W, float *F_out, bool *M,
                  unsigned int b, unsigned int h, unsigned int w,
                  unsigned int c_in, unsigned int c_out,
                  unsigned int f, unsigned int s, unsigned int p) {
    assert(h == w);

    // Moved to main() so that it's only done once
    //falcon_init_lib();

    unsigned int MM = 1;
    float *image = F_in;
    //unsigned int irows = h;
    unsigned int irows = h+2*p;
    unsigned int C = c_in;
    float *filter = W;
    unsigned int K = c_out; 
    unsigned int batch = b; 
    float *out = F_out; 

    fal_conv(MM,image,irows,C,filter,K,batch,out);

    //falcon_free_lib();
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

template <unsigned int c_in, unsigned int c_out, unsigned int h, unsigned int w>
void do_one_layer(unsigned int layer, float sparsity) {
    assert(h == w);
    const unsigned int f = 3;
    //const unsigned int c_in = 64;
    //const unsigned int c_out = 128;
    //const unsigned int h = 128;
    //const unsigned int w = 128;
    const unsigned int p = 1;
    const unsigned int b = 1;
    const unsigned int s = 1;
    unsigned int h_out = ((h - f + 2*p)/s + 1);
    unsigned int w_out = ((w - f + 2*p)/s + 1);

    float *F_in = (float*) aligned_alloc(64, sizeof(float) * b * (h + 2*p) * (w + 2*p) * c_in);
    float *W = (float*) aligned_alloc(64, sizeof(float) * c_out * c_in * f * f);
    float *F_out = (float*) aligned_alloc(64, sizeof(float) * b * h_out * w_out * c_out);
    bool *M = (bool*) aligned_alloc(64, sizeof(bool) *b * h_out * w_out);

    generate_random_tensor(F_in, b, h + 2*p, w + 2*p, c_in);
    generate_random_tensor(W, c_in, f, f, c_out);
    generate_random_tensor(F_out, b, h_out, w_out, c_out);
    unsigned int nnz = generate_sparsity_pattern(M, b, h_out, w_out, sparsity);

    float time_dense = benchmark(5, 1, [&]() {
        conv_naive<b, h, w, c_in, c_out, f, s, p>(F_in, W, F_out, nullptr);
    });

    float gfops_dense = ((float)b * c_in * h_out * w_out * c_out * f * f)/(1e09);
    float gflops_dense = (gfops_dense)/time_dense;
    //std::cout << "Dense Time : " << 1000 * time_dense << "ms " << std::endl;
    //std::cout << "GFLOPS : " << gflops_dense << std::endl;

    float time_dense_tmpl = benchmark(5, 1, [&]() {
        conv_naive_tmpl<b, h, w, c_in, c_out, f, s, p>(F_in, W, F_out, nullptr);
    });

    float gflops_dense_tmpl = (gfops_dense)/time_dense_tmpl;
    //std::cout << "Dense Template Time : " << 1000 * time_dense_tmpl << "ms " << std::endl;
    //std::cout << "GFLOPS : " << gflops_dense_tmpl << std::endl;

    const unsigned int tile_h = 4;
    const unsigned int tile_w = 8;
    float time_tiled_spatial = benchmark(5, 1, [&]() {
        conv_tiled_spatial<b, h, w, c_in, c_out, f, s, p, tile_h, tile_w>(F_in, W, F_out, nullptr);
    });

    float gflops_tiled_spatial = (gfops_dense)/time_tiled_spatial;
    //std::cout << "Dense Spatial Tile Time : " << 1000 * time_tiled_spatial << "ms " << std::endl;
    //std::cout << "GFLOPS : " << gflops_tiled_spatial << std::endl;

    const unsigned int t1 = 8;
    const unsigned int t2 = 16;
    float time_tiled_3d = benchmark(5, 1, [&]() {
        conv_tiled_3d<b, h, w, c_in, c_out, f, s, p, t1, t2>(F_in, W, F_out, nullptr);
    });

    float gflops_tiled_3d = (gfops_dense)/time_tiled_3d;
    //std::cout << "Dense 3D Tile Time : " << 1000 * time_tiled_3d << "ms " << std::endl;
    //std::cout << "GFLOPS : " << gflops_tiled_3d << std::endl;

    float time_sparse = benchmark(5, 1, [&]() {
        conv_naive<b, h, w, c_in, c_out, f, s, p>(F_in, W, F_out, M);
    });

    float gfops_sparse = ((float)b * c_in * nnz * c_out * f * f)/(1e09);
    float gflops_sparse = (gfops_sparse)/time_sparse;
    //std::cout << "Sparse Time: " << 1000 * time_sparse << "ms " << std::endl;
    //std::cout << "GFLOPS : " << gflops_sparse << std::endl;

    float time_winograd = benchmark(5, 1, [&]() {
        conv_winograd(F_in, W, F_out, M, b, h, w, c_in, c_out, f, s, p);
    });
    float gflops_winograd = (gfops_dense)/time_winograd;
    //std::cout << "Winograd Time: " << 1000 * time_winograd << "ms " << std::endl;
    //std::cout << "GFLOPS : " << gflops_winograd << std::endl;

    //std::cout << layer << "," << gflops_dense << "," << gflops_dense_tmpl << "," << gflops_tiled_spatial << "," <<gflops_tiled_3d << "," <<gflops_sparse << "," << gflops_winograd << std::endl;
    std::cout << layer << "," << 1000*time_dense << "," << 1000*time_dense_tmpl<< "," << 1000*time_tiled_spatial<< "," <<1000*time_tiled_3d<< "," <<1000*time_sparse<< "," << 1000*time_winograd<< std::endl;

    free(F_in);
    free(W);
    free(F_out);
    free(M);

}

int main(int argc, char *argv[]) {
    const unsigned int threads[6] = {1, 2, 4, 8, 16, 32};
    const float sparsity_array[4] = {0.05, 0.1, 0.25, 0.5};
    const unsigned int hw_array[13] =   {226, 226, 114, 114,  58,  58,  58,  30,  30,  30,  16,  16,  16};
    const unsigned int cin_array[13] =  {3,    64,  64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512};
    const unsigned int cout_array[13] = {64,   64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512};
    //const int merge_array[13] = {1, 1, 4, 4, 8, 8, 8, 16, 16, 16, 16, 16, 16};

    falcon_init_lib();

    int bench = (argc > 1 ? atoi(argv[1]) : 0);

    if (bench == 0) { 
        for (unsigned int s = 0; s < 4; s++) {
            float sparsity = sparsity_array[s];
            std::cout << "----------------------- Sparsity - " << sparsity << " -----------------------" << std::endl;
            do_one_layer<  3,  64, SCALE(226), SCALE(226)>(1, sparsity);
            do_one_layer< 64,  64, SCALE(226), SCALE(226)>(2, sparsity);
            do_one_layer< 64, 128, SCALE(114), SCALE(114)>(3, sparsity);
            do_one_layer<128, 128, SCALE(114), SCALE(114)>(4, sparsity);
            do_one_layer<128, 256,  SCALE(58),  SCALE(58)>(5, sparsity);
            do_one_layer<256, 256,  SCALE(58),  SCALE(58)>(6, sparsity);
            do_one_layer<256, 256,  SCALE(58),  SCALE(58)>(7, sparsity);
            do_one_layer<256, 512,  SCALE(30),  SCALE(30)>(8, sparsity);
            do_one_layer<512, 512,  SCALE(30),  SCALE(30)>(9, sparsity);
            do_one_layer<512, 512,  SCALE(30),  SCALE(30)>(10, sparsity);
            do_one_layer<512, 512,  SCALE(16),  SCALE(16)>(11, sparsity);
            do_one_layer<512, 512,  SCALE(16),  SCALE(16)>(12, sparsity);
            do_one_layer<512, 512,  SCALE(16),  SCALE(16)>(13, sparsity);
        }
    } else {
        float sparsity = 0.1;
	for (unsigned int t = 0; t < 6; t++) {
	    unsigned int num_threads = threads[t];
            std::cout << "----------------------- Threads - " << num_threads << " -----------------------" << std::endl;
	    omp_set_num_threads(num_threads);
            do_one_layer<  3,  64, SCALE(226), SCALE(226)>(1, sparsity);
            do_one_layer< 64,  64, SCALE(226), SCALE(226)>(2, sparsity);
            do_one_layer< 64, 128, SCALE(114), SCALE(114)>(3, sparsity);
            do_one_layer<128, 128, SCALE(114), SCALE(114)>(4, sparsity);
            do_one_layer<128, 256,  SCALE(58),  SCALE(58)>(5, sparsity);
            do_one_layer<256, 256,  SCALE(58),  SCALE(58)>(6, sparsity);
            do_one_layer<256, 256,  SCALE(58),  SCALE(58)>(7, sparsity);
            do_one_layer<256, 512,  SCALE(30),  SCALE(30)>(8, sparsity);
            do_one_layer<512, 512,  SCALE(30),  SCALE(30)>(9, sparsity);
            do_one_layer<512, 512,  SCALE(30),  SCALE(30)>(10, sparsity);
            do_one_layer<512, 512,  SCALE(16),  SCALE(16)>(11, sparsity);
            do_one_layer<512, 512,  SCALE(16),  SCALE(16)>(12, sparsity);
            do_one_layer<512, 512,  SCALE(16),  SCALE(16)>(13, sparsity);
	}
    }

#if 0
    for (unsigned int s = 0; s < 4; s++) {
        float sparsity = sparsity_array[s];
        for (unsigned layer = 0; layer < 13; layer++){
            unsigned int h = hw_array[layer];
            unsigned int w = h;
            unsigned int c_in = cin_array[layer];
            unsigned int c_out = cout_array[layer];

            do_one_layer<c_in, c_out, h, w>(layer, sparsity);
        }
    }
#endif

    falcon_free_lib();

    return 0;
}
