#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols == b.cols);

    matrix y = copy_matrix(xw);
    int i,j;
    for(i = 0; i < xw.rows; ++i){
        for(j = 0; j < xw.cols; ++j){
            y.data[i*y.cols + j] += b.data[j];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_bias(matrix dy)
{
    matrix db = make_matrix(1, dy.cols);
    int i, j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = xw+b
matrix forward_connected_layer(layer l, matrix x)
{


    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);



    matrix wx = matmul(x, l.w);
    matrix y = forward_bias(wx, l.b);

    free_matrix(wx);
    
    return y;
}

// Run a connected layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_connected_layer(layer l, matrix dy)
{
    matrix x = *l.x;

    // TODO: 3.2
    // Calculate the gradient dL/db for the bias terms using backward_bias
    // add this into any stored gradient info already in l.db
    matrix db = backward_bias(dy);
    axpy_matrix(1, db, l.db);

    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw
    matrix xt = transpose_matrix(x);
    matrix dw = matmul(xt, dy);
    axpy_matrix(1, dw, l.dw);


    // Calculate dL/dx and return it
    // matrix dx = copy_matrix(x); // Change this
    matrix wt = transpose_matrix(l.w);
    matrix dx = matmul(dy, wt);

    free_matrix(dw);
    free_matrix(db);
    free_matrix(xt);
    free_matrix(wt);


    return dx;
}

// Update weights and biases of connected layer
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_connected_layer(layer l, float rate, float momentum, float decay)
{
    if (l.freeze == 0){

    // Apply our updates using our SGD update rule
    // assume  l.dw = dL/dw - momentum * update_prev
    // we want l.dw = dL/dw - momentum * update_prev + decay * w
    // then we update l.w = l.w - rate * l.dw
    // lastly, l.dw is the negative update (-update) but for the next iteration
    // we want it to be (-momentum * update) so we just need to scale it a little
    axpy_matrix(decay, l.w, l.dw);
    axpy_matrix(-rate, l.dw, l.w);
    scal_matrix(momentum, l.dw);
    // Do the same for biases as well but no need to use weight decay on biases
    axpy_matrix(-rate, l.db, l.b);
    scal_matrix(momentum, l.db);
    }
}

layer make_connected_layer(int inputs, int outputs)
{
    layer l = {0};
    l.w  = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
    l.dw = make_matrix(inputs, outputs);
    l.b  = make_matrix(1, outputs);
    l.db = make_matrix(1, outputs);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    return l;
}

