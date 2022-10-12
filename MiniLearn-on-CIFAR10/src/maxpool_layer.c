#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer

matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int w = ((l.width - l.size)/l.stride )+ 1;
    int h = ((l.height - l.size) / l.stride )+ 1;
    int c = l.channels;
    
    matrix out = make_matrix(in.rows, w*h*c);

   int i, j, k, m, n, b;

   for(b = 0; b < in.rows; ++b){
        for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                    float max = -FLT_MAX;
                    int max_i = 0;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h =  i*l.stride + n;
                            int cur_w =  j*l.stride + m;
                            int index = cur_w + l.width*(cur_h + l.height*(k + b*l.channels));
                            int valid = (cur_h >= 0 && cur_h < l.height &&
                                         cur_w >= 0 && cur_w < l.width);
                            if(index > in.rows *in.cols)
                                printf("index out of matrix \n");
                            float val = (valid != 0) ? in.data[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    int out_index = b * out.cols + k*h*w + k*w + j;
                    if(out_index > out.cols *out.rows){
                        printf("out inex out of indx , \n");
                    }
                    out.data[out_index] = max;
                }
            }
        }
    }
    return out;
}


// matrix backward_maxpool_layer(layer l, matrix dy)
// {

//     matrix in    = *l.x;
//     matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);


//     int i;
//     int w = ((l.width - l.size)/l.stride )+ 1;
//     int h = ((l.height - l.size) / l.stride )+ 1;

//     int c = l.channels;

//     for(i = 0; i < h*w*c*in.rows; ++i){
//         int index = l.indexes[i];
//         dx.data[index] = dx.data[index] + dy.data[i];
//     }

//     return dx;
// }



matrix backward_maxpool_layer(layer l, matrix dy)
{
    int i, c, h, w;
    int fr, fc;
    int outr, outc;

    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);


    int outw = ((l.width - l.size)/l.stride )+ 1;
    int outh = ((l.height - l.size) / l.stride )+ 1;
    
    for (i = 0; i < in.rows; i++) {
        // get 1 image
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        
        // loop over original image by channel, rows, then collumn
        for (c = 0; c < example.c; c++) {
            for (h = 0, outr = 0; h < example.h; h += l.stride, outr++) {
                for (w = 0, outc = 0; w < example.w; w += l.stride, outc++) {
                    float max = FLT_MIN;
                    int idxr = 0;
                    int idxc = 0;
                    float val;     
                    // loop over the maxpool filter
                    for (fr = 0; fr < l.size; fr++) {
                        for (fc = 0; fc < l.size; fc++) {
                            if (w + fc >= 0 && h + fr >= 0 && w + fc < example.w && h + fr < example.h) {
                                val = get_pixel(example, w + fc, h + fr, c);
                                if (val > max) {
                                    max = val;
                                    idxr = fr;
                                    idxc = fc;
                                }
                            }
                        }
                    }
                    val = get_matrix(dy, i, (outh * outw * c) + (outw * outr) + outc);
                    set_matrix(dx, i, (l.height * l.width * c) + (l.width * (h + idxr)) + (w + idxc), val);
                }
            }
        }
    }

    return dx;
}


// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){
    (void) l;
    (void)rate;
    (void)momentum;
    (void)decay;
}


// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;

    // int out_w = (w + 2*padding)/stride;
    // int out_h = (h + 2*padding)/stride;


    // int w_out = ((l.width - l.size)/l.stride )+ 1;
    // int h_out = ((l.height - l.size) / l.stride )+ 1;
    // int output_size = h_out * w_out * c *4 ;
    // l.indexes = calloc(output_size, sizeof(int));

    return l;
}






// matrix forward_maxpool_layer(layer l, matrix in)
// {
//     // Saving our input
//     // Probably don't change this
//     free_matrix(*l.x);
//     *l.x = copy_matrix(in);

//     int i, c, h, w;
//     int fr, fc;
//     int outr, outc;
//     // int outw = (l.width-1)/l.stride + 1;
//     // int outh = (l.height-1)/l.stride + 1;
//     int outw = ((l.width - l.size)/l.stride )+ 1;
//     int outh = ((l.height - l.size) / l.stride )+ 1;

//     matrix out = make_matrix(in.rows, outw*outh*l.channels);
//     float val;

//     // TODO: 6.1 - iterate over the input and fill in the output with max values
//     for (i = 0; i < in.rows; i++) {
//         // get 1 image
//         image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        
//         // loop over original image by channel, rows, then collumn
//         for (c = 0; c < example.c; c++) {
//             for (h = 0, outr = 0; h < example.h; h += l.stride, outr++) {
//                 for (w = 0, outc = 0; w < example.w; w += l.stride, outc++) {
//                     float max = FLT_MIN;
                    
//                     // loop over the maxpool filter
//                     for (fr = 0; fr < l.size; fr++) {
//                         for (fc = 0; fc < l.size; fc++) {
//                             if (w + fc >= 0 && h + fr >= 0 && w + fc < example.w && h + fr < example.h) {
//                                 val = get_pixel(example, w + fc, h + fr, c);
//                                 if (val > max) { max = val;}
//                             }
//                         }
//                     }

//                     set_matrix(out, i, (outh * outw * c) + (outw * outr) + outc, max);
//                 }
//             }
//         }
//     }


//     return out;
// }

// // Run a maxpool layer backward
// // layer l: layer to run
// // matrix prev_delta: error term for the previous layer