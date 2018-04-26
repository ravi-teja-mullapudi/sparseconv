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

// F_in is a tensor of dimension b x (h+p) x (w+p) x c_in
// F_out is a tensor of dimension b x ((h - f + 2p)/s + 1) x ((w - f + 2p)/s + 1) x c_out
// W is a tensor of dimension c_in x c_out x f x f
int conv_naive(float &F_in, float &W, float &F_out, bool &M,
               size_t b, size_t h, size_t w, size_t c_in, size_t c_out,
               size_t f, size_t s, size_t p) {

}

// F_in is a tensor of dimension b x (h+p) x (w+p) x c_in
// F_out is a tensor of dimension b x ((h - f + 2p)/s + 1) x ((w - f + 2p)/s + 1) x c_out
// W is a tensor of dimension c_in x c_out x f x f
// M is a tensor of dimension b x ((h - f + 2p)/s + 1) x ((w - f + 2p)/s + 1) x c_out
int sparse_conv_naive(float &F_in, float &W, float &F_out, bool &M,
                      size_t b, size_t h, size_t w, size_t c_in, size_t c_out) {
}
