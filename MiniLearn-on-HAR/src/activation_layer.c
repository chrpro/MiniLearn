#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"


// Run an activation layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = f(x)
matrix forward_activation_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    ACTIVATION a = l.activation;
    matrix y = copy_matrix(x);

    // TODO: 2.1
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row 
    int i, j;
    for(i = 0; i < x.rows; ++i){
        float sum = 0;
        for(j = 0; j < x.cols; ++j){
            int index = i*x.cols + j;
            float v = x.data[index];
            if(a == LOGISTIC){
                y.data[index] = 1/(1+expf(-v));
            } else if (a == RELU){
                y.data[index] = (v>0)*v;
            } else if (a == LRELU){
                y.data[index] = (v>0) ? v : .01*v;
            } else if (a == SOFTMAX){
                y.data[index] = expf(v);
            }
            sum += y.data[index];
        }
        if (a == SOFTMAX) {
            for(j = 0; j < x.cols; ++j){
                int index = i*x.cols + j;
                y.data[index] /= sum;
            }
        }
    }

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_activation_layer(layer l, matrix dy)
{
    matrix x = *l.x;
    matrix dx = copy_matrix(dy);
    ACTIVATION a = l.activation;

    // TODO: 2.2
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1

    int i, j;
    for(i = 0; i < dx.rows; ++i){
        for(j = 0; j < dx.cols; ++j){
            float v = x.data[i*x.cols + j];
            if(a == LOGISTIC){
                float fx = 1/(1 + exp(-v));
                dx.data[i*dx.cols + j] *= fx*(1-fx);
            } else if (a == RELU){
                dx.data[i*dx.cols + j] *= (v>0) ? 1 : 0;
            } else if (a == LRELU){
                dx.data[i*dx.cols + j] *= (v>0) ? 1 : 0.01;
            }
        }
    }

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
