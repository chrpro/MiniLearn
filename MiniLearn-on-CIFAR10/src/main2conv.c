//Standart libs
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

//Training libs
#include "uwnet.h"
#include "test.h"
#include "args.h"

// CMSIS Libs
#include "arm_math.h"
#include "arm_nnfunctions.h"

//Train images
#include "cifar_train_data.h"
// Test images
#include "cifar_test_data.h"

// Neural Network
#include "parameters.h"
#include "weights.h"

# define CLASSES 3


// static  int numberOFilters = 12 ;// ->CONV3_OUT_CH
# define numberOFilters 24
# define numberOFilters2 32
// # define numberOFiltersConv1 32

static q7_t conv3_w_2[numberOFilters*CONV3_IM_CH*CONV3_KER_DIM*CONV3_KER_DIM] =  {0};
static q7_t conv3_b_2[numberOFilters] = {0};
static q7_t conv3_out[numberOFilters*CONV3_OUT_DIM*CONV3_OUT_DIM];
static q7_t pool3_out[numberOFilters*POOL3_OUT_DIM*POOL3_OUT_DIM];

static q7_t conv2_w_2[numberOFilters2*CONV2_IM_CH*CONV2_KER_DIM*CONV2_KER_DIM] =  {0};
static q7_t conv2_b_2[numberOFilters2] = {0};
static q7_t conv2_out_2[numberOFilters2*CONV2_OUT_DIM*CONV2_OUT_DIM];
static q7_t pool2_out_2[numberOFilters2*POOL2_OUT_DIM*POOL2_OUT_DIM];


static q7_t interface_out[INTERFACE_OUT];
static q7_t linear_out[LINEAR_OUT];
static q7_t y_out[INTERFACE_OUT];

uint32_t network(q7_t* input);
uint32_t network2(q7_t* input);


static q7_t conv1_out[CONV1_OUT_CH*CONV1_OUT_DIM*CONV1_OUT_DIM];
static q7_t pool1_out[CONV1_OUT_CH*POOL1_OUT_DIM*POOL1_OUT_DIM];
static q7_t conv2_out[CONV2_OUT_CH*CONV2_OUT_DIM*CONV2_OUT_DIM];
static q7_t pool2_out[CONV2_OUT_CH*POOL2_OUT_DIM*POOL2_OUT_DIM];

static q7_t conv1_w[CONV1_WT_SHAPE] = CONV1_WT;
static q7_t conv1_b[CONV1_BIAS_SHAPE] = CONV1_BIAS;
static q7_t conv2_w[CONV2_WT_SHAPE] =  CONV2_WT;
static q7_t conv2_b[CONV2_BIAS_SHAPE] = CONV2_BIAS;
static q7_t conv3_w[CONV3_WT_SHAPE] =  CONV3_WT;
static q7_t conv3_b[CONV3_BIAS_SHAPE] = CONV3_BIAS;



static q15_t conv_buffer[MAX_CONV_BUFFER_SIZE];
static q15_t fc_buffer[MAX_FC_BUFFER];


static q7_t interface_w[INTERFACE_WT_SHAPE] = INTERFACE_WT;
static q7_t interface_b[INTERFACE_BIAS_SHAPE] = INTERFACE_BIAS;
static q7_t linear_w[LINEAR_WT_SHAPE] = LINEAR_WT;
static q7_t linear_b[LINEAR_BIAS_SHAPE] = LINEAR_BIAS;




static int batch = 4;



int main(void)
{
    const char chalmersBanner[] = {
		"============================================================\n"
		"Tranfers Learning on low-power IoT devices                  \n"
		"Chalmers University                                         \n"
		"============================================================\n"
	}; printf(chalmersBanner);  




    struct timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    // FILE* fp ;
    // fp = fopen("test-accuracies-scale.csv", "w"); // save result in csv file
    // fprintf(fp, "Epochs, Test Accuracy\n");	

    uint32_t index = 0;
    int PoolOut = POOL1_OUT_DIM* POOL1_OUT_DIM*CONV1_OUT_CH;
    
    // int fc_layer_out = 160;

    float rate = .0001;
    float momentum = .9;
    float decay = .00001;
    int correct = 0;

    data training_data;
    int sub_sample = 95;

    matrix q_X0 = make_matrix(sub_sample,PoolOut);
    matrix q_Y0 = make_matrix(sub_sample,CLASSES);
    // float scale = 1 << CONV1_OUT_Q;

    for(int img_row=0; img_row < sub_sample; img_row++)
    {
        index = network(NODE_0_TRAIN_IMAGES[img_row]);


        for (int i = 0; i < PoolOut; i++){
            q_X0.data[ img_row * q_X0.cols + i] = (float) pool1_out[i] /  powf(2,CONV1_OUT_Q) ;        
        }
        q_Y0.data[ (img_row * q_Y0.cols) + NODE_0_TRAIN_LABELS[img_row] ] = 1;     
    }

    training_data.x= q_X0;
    training_data.y = q_Y0;

    matrix X_test = make_matrix(TOTAL_TEST_IMAGES,PoolOut);
    matrix Y_test = make_matrix(TOTAL_TEST_IMAGES,CLASSES);

    for(int img_i=0; img_i < TOTAL_TEST_IMAGES; img_i++)
    {
        index = network(TEST_IMAGES[img_i]);
        if((uint32_t)TEST_LABELS[img_i] == index){
            correct++;
    
        }
        for (int i = 0; i < PoolOut; i++){
            X_test.data[ img_i*X_test.cols + i] = (float) pool1_out[i] /  powf(2,CONV1_OUT_Q) ;
        }
        Y_test.data[ (img_i*Y_test.cols) + TEST_LABELS[img_i] ] = 1;      
    }
    printf("=> CMSIS accuracy: %f\n", (float)correct / TOTAL_TEST_IMAGES );

    data test;
    test.x=X_test;
    test.y=Y_test;

    int iterations = sub_sample/ batch;
    // printf("iterations = %d\n",iterations);
    // int convout = CONV3_OUT_CH*CONV3_OUT_DIM*CONV3_OUT_DIM;

    static net active_learning = {0}; 
    active_learning.n = 10;
    active_learning.layers = calloc(10, sizeof(layer));   

    int filterSize1 =  CONV2_IM_CH*CONV2_KER_DIM*CONV2_KER_DIM;
    
    active_learning.layers[0] = make_convolutional_layer(CONV2_IM_DIM, CONV2_IM_DIM, CONV2_IM_CH, numberOFilters2, 
        CONV2_KER_DIM , CONV2_STRIDE);
    
    // float scale2 = 1 << CONV3_OUT_Q;

    for (int index = 0; index < filterSize1 * numberOFilters2; index++)
    {
        active_learning.layers[0].w.data[index] = (float) conv2_w[index] /  powf(2, CONV2_WEIGHT_Q);
    }
    
    for (int index = 0; index < filterSize1; index++)
    {
        active_learning.layers[0].b.data[index] = (float)  conv2_b[index]/  powf(2, CONV2_BIAS_Q);
    }

    active_learning.layers[1] = make_batchnorm_layer(numberOFilters2);
    active_learning.layers[2] = make_activation_layer(RELU);
    
    // active_learning.layers[2] = make_maxpool_layer( POOL2_IM_DIM, POOL2_IM_DIM, numberOFilters2,  POOL2_KER_DIM , POOL2_STRIDE); 

    // int outw = ((POOL2_IM_DIM  - POOL2_KER_DIM)/POOL2_STRIDE )  + 1;
    // int outh = ((l.height - l.size)/l.stride ) + 1;

    //filter size 32*3*3
    int filterSize =  CONV3_IM_CH*CONV3_KER_DIM*CONV3_KER_DIM;
    // number of filters 16
    // make it 8
    // int numberOFilters = 12 ;// ->CONV3_OUT_CH

    // active_learning.layers[0] = make_convolutional_layer(CONV3_IM_DIM, CONV3_IM_DIM, CONV3_IM_CH, CONV3_OUT_CH, CONV3_KER_DIM , CONV3_STRIDE);

    // active_learning.layers[3] = make_convolutional_layer(outw, outw, numberOFilters2, numberOFilters, CONV3_KER_DIM , CONV3_STRIDE);
    active_learning.layers[3] = make_convolutional_layer(POOL2_IM_DIM, POOL2_IM_DIM, numberOFilters2, numberOFilters, CONV3_KER_DIM , CONV3_STRIDE);
        


    // for (int index = 0; index < CONV3_WT_SHAPE; index++)

    for (int index = 0; index < filterSize * numberOFilters; index++)
    {
        active_learning.layers[3].w.data[index] = (float) (conv3_w[index]  ) /  powf(2, CONV3_WEIGHT_Q); 
        // printf("%f, ",active_learning.layers[4].w.data[index]);
    }
    for (int index = 0; index < filterSize; index++)
    {
        active_learning.layers[3].b.data[index] = (float)  conv3_b[index] /  powf(2, CONV3_BIAS_Q);
        // printf("%f, ",active_learning.layers[0].b.data[index]);
    }
    
    active_learning.layers[4] = make_batchnorm_layer(numberOFilters);

    active_learning.layers[5] = make_activation_layer(RELU);
    // active_learning.layers[2] = make_maxpool_layer( POOL3_IM_DIM, POOL3_IM_DIM, POOL3_IM_CH,  POOL3_KER_DIM , POOL3_STRIDE); 
    
    // int convout = numberOFilters*POOL3_OUT_DIM*POOL3_OUT_DIM;
    // int convout = numberOFilters*CONV3_OUT_DIM*CONV3_OUT_DIM;
    int outw2 = ((POOL2_IM_DIM  - CONV3_KER_DIM)/CONV3_STRIDE )  + 1;
    int convout = numberOFilters*outw2*outw2;
// 

    // printf("CONECTED IN %d\n" ,convout );
  
    active_learning.layers[6] = make_connected_layer(convout, INTERFACE_OUT); 
    
    // for (int index = 0; index < INTERFACE_WT_SHAPE; index++)
    // for (int index = 0; index < convout*INTERFACE_OUT; index++)
    // {
    //     active_learning.layers[4].w.data[index] = (float) interface_w[index] /  powf(2, INTERFACE_WEIGHT_Q);
    // }
    
    // for (int index = 0; index < INTERFACE_BIAS_SHAPE; index++)
    // {
    //     active_learning.layers[4].b.data[index] = (float)  interface_b[index] /  powf(2, INTERFACE_BIAS_Q);
    // }
    
    active_learning.layers[7] = make_activation_layer(RELU);
    active_learning.layers[8] = make_connected_layer(INTERFACE_OUT , CLASSES);
    active_learning.layers[9] = make_activation_layer(SOFTMAX);

    iterations = 80;
    for ( int epoch = 1; epoch < 3; epoch++){
        train_image_classifier(active_learning, training_data, batch, iterations,  rate, momentum, decay);
        
        float train_acc = accuracy_net(active_learning, training_data);
        printf("%d :: train acc = %f & ",epoch, train_acc);
        float test_acc = accuracy_net(active_learning, test);
        printf("test acc = %f\n", test_acc);

    }


    printf("-----------------------------------------------------------------------------\n");

    for (int index = 0; index < filterSize * numberOFilters2; index++)
    {
        // active_learning.layers[0].w.data[index] = conv3_w_float [index] ;
        conv2_w_2[index] = (int) ceilf (active_learning.layers[0].w.data[index] *  powf(2, CONV2_WEIGHT_Q) );
        // printf("%d, ",conv3_w[index]);
    }


    for (int index = 0; index < numberOFilters; index++)
    {
        conv2_b_2[index] = (int) ceilf ( active_learning.layers[0].b.data[index] * powf(2, CONV2_BIAS_Q) );
        
        // printf("%d, ",conv3_b[index]);
    }

    // for (int index = 0; index < CONV3_WT_SHAPE; index++)
    for (int index = 0; index < filterSize * numberOFilters; index++)
    {
        // active_learning.layers[0].w.data[index] = conv3_w_float [index] ;
        conv3_w_2[index] = (int) ceilf (active_learning.layers[3].w.data[index] *  powf(2, CONV3_WEIGHT_Q) );
        // printf("%d, ",conv3_w[index]);
    }


    for (int index = 0; index < numberOFilters; index++)
    {
        conv3_b_2[index] = (int) ceilf ( active_learning.layers[3].b.data[index] * powf(2, CONV3_BIAS_Q) );
    }


    PoolOut = numberOFilters*CONV3_OUT_DIM*CONV3_OUT_DIM;
    // PoolOut = CONV3_OUT_DIM* CONV3_OUT_DIM*numberOFilters;
    // printf("2CONECTED IN %d\n" ,PoolOut );
    active_learning.n = 5;
    active_learning.layers[0] = make_activation_layer(RELU);  //active_learning.layers[1];
    active_learning.layers[1] = make_connected_layer(PoolOut, INTERFACE_OUT);
    active_learning.layers[2] = make_activation_layer(RELU);// active_learning.layers[3];
    active_learning.layers[3] = make_connected_layer(INTERFACE_OUT , CLASSES);
    active_learning.layers[4] = make_activation_layer(SOFTMAX);// active_learning.layers[5];

    // for (int index = 0; index < PoolOut*INTERFACE_OUT; index++)
    // {
    //     active_learning.layers[3].w.data[index] = (float) interface_w[index] /  powf(2, INTERFACE_WEIGHT_Q);
    // }
    
    // for (int index = 0; index < INTERFACE_BIAS_SHAPE; index++)
    // {
    //     active_learning.layers[3].b.data[index] = (float)  interface_b[index] /  powf(2, INTERFACE_BIAS_Q);
    // }



    // active_learning.layers[0] =  active_learning.layers[2];
    // active_learning.layers[2] =  active_learning.layers[4];

    // active_learning.layers[4] =  active_learning.layers[5];



    data training_data2;
    // PoolOut = POOL3_OUT_DIM* POOL3_OUT_DIM*numberOFilters;
  
    matrix q_X = make_matrix(NODE_0_TOTAL_TRAIN_IMAGES,PoolOut);
    matrix q_Y = make_matrix(NODE_0_TOTAL_TRAIN_IMAGES,CLASSES);

    // float scale = 1 << CONV1_OUT_Q;

    for(int img_row=0; img_row < NODE_0_TOTAL_TRAIN_IMAGES; img_row++)
    {
        index = network2(NODE_0_TRAIN_IMAGES[img_row]);
        for (int i = 0; i < PoolOut; i++){
            q_X.data[ img_row * q_X.cols + i] = (float) conv3_out[i] /  powf(2,CONV3_OUT_Q) ;        
        }
        q_Y.data[ (img_row * q_Y.cols) + NODE_0_TRAIN_LABELS[img_row] ] = 1;     
    }
    
    training_data2.x= q_X;
    training_data2.y = q_Y;


    matrix X_test2 = make_matrix(TOTAL_TEST_IMAGES,PoolOut);
    matrix Y_test2 = make_matrix(TOTAL_TEST_IMAGES,CLASSES);

    for(int img_i=0; img_i < TOTAL_TEST_IMAGES; img_i++)
    {
        index = network2(TEST_IMAGES[img_i]);
        for (int i = 0; i < PoolOut; i++){
            X_test2.data[ img_i*X_test2.cols + i] = (float) conv3_out[i] /  powf(2,CONV3_OUT_Q) ;
        }
        Y_test2.data[ (img_i*Y_test2.cols) + TEST_LABELS[img_i] ] = 1;      
    }

    data test2;
    test2.x=X_test2;
    test2.y=Y_test2;


    float test_acc = accuracy_net(active_learning, test2);
    printf("test acc = %f\n", test_acc);

    for ( int epoch = 1; epoch < 25; epoch++){
        train_image_classifier(active_learning, training_data2, batch, iterations,  rate, momentum, decay);
        float train_acc = accuracy_net(active_learning, training_data2);
        printf("%d :: train acc = %f & ",epoch, train_acc);
        float test_acc = accuracy_net(active_learning, test2);
        printf("test acc = %f\n", test_acc);

    }


    return 0;
}




uint32_t network(q7_t* input)
{


    	arm_convolve_HWC_q7_RGB(input, CONV1_IM_DIM, CONV1_IM_CH, conv1_w, 
                        CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, 
                        conv1_b, CONV1_BIAS_LSHIFT, 
                          CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						  conv_buffer, NULL);

        arm_maxpool_q7_HWC(conv1_out, POOL1_IM_DIM, POOL1_IM_CH, POOL1_KER_DIM, POOL1_PADDING,
         POOL1_STRIDE, POOL1_OUT_DIM, (q7_t *) conv_buffer, pool1_out);
        arm_relu_q7(pool1_out, POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH);

        arm_convolve_HWC_q7_fast(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, conv2_w, CONV2_OUT_CH, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, conv2_b, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM, conv_buffer, NULL);


        arm_maxpool_q7_HWC(conv2_out, POOL2_IM_DIM, POOL2_IM_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, (q7_t *) conv_buffer, pool2_out);
        arm_relu_q7(pool2_out, POOL2_OUT_DIM * POOL2_OUT_DIM * CONV2_OUT_CH);



        arm_convolve_HWC_q7_fast(pool2_out, CONV3_IM_DIM, CONV3_IM_CH, conv3_w, CONV3_OUT_CH, CONV3_KER_DIM,
						  CONV3_PADDING, CONV3_STRIDE, conv3_b, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
						  CONV3_OUT_DIM, conv_buffer, NULL);
        arm_maxpool_q7_HWC(conv3_out, POOL3_IM_DIM, POOL3_IM_CH, POOL3_KER_DIM, POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM,  (q7_t *) conv_buffer, pool3_out);
        arm_relu_q7(pool3_out, POOL3_OUT_DIM * POOL3_OUT_DIM * CONV3_OUT_CH); 
        	arm_fully_connected_q7_opt(pool3_out, interface_w, INTERFACE_DIM, INTERFACE_OUT, INTERFACE_BIAS_LSHIFT, INTERFACE_OUT_RSHIFT, interface_b,
						   interface_out, fc_buffer);
    
	arm_relu_q7(interface_out, INTERFACE_OUT);

	
	arm_fully_connected_q7_opt(interface_out, linear_w, LINEAR_DIM, LINEAR_OUT, LINEAR_BIAS_LSHIFT, LINEAR_OUT_RSHIFT, linear_b,
						  linear_out, fc_buffer);
    

	
    arm_softmax_q7(linear_out, LINEAR_OUT, y_out);
	uint32_t index[1];
	q7_t result[1];
	uint32_t blockSize = sizeof(y_out);

	arm_max_q7(y_out, blockSize, result, index);

      return index[0];
}




uint32_t network2(q7_t* input)
{


    	arm_convolve_HWC_q7_RGB(input, CONV1_IM_DIM, CONV1_IM_CH, conv1_w, 
                        CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, 
                        conv1_b, CONV1_BIAS_LSHIFT, 
                          CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						  conv_buffer, NULL);

        arm_maxpool_q7_HWC(conv1_out, POOL1_IM_DIM, POOL1_IM_CH, POOL1_KER_DIM, POOL1_PADDING,
         POOL1_STRIDE, POOL1_OUT_DIM, (q7_t *) conv_buffer, pool1_out);
        arm_relu_q7(pool1_out, POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH);

        arm_convolve_HWC_q7_fast(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, conv2_w_2, numberOFilters2, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, conv2_b_2, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM, conv_buffer, NULL);

        //arm_maxpool_q7_HWC(conv2_out, POOL2_IM_DIM, POOL2_IM_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, (q7_t *) conv_buffer, pool2_out);
        // arm_relu_q7(pool2_out, POOL2_OUT_DIM * POOL2_OUT_DIM * CONV2_OUT_CH);
        arm_relu_q7(conv2_out, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);

//////// conv 3 
        arm_convolve_HWC_q7_fast(conv2_out, POOL2_IM_DIM, numberOFilters2, conv3_w_2, numberOFilters, CONV3_KER_DIM,
						  CONV3_PADDING, CONV3_STRIDE, conv3_b_2, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
						  CONV3_OUT_DIM, conv_buffer, NULL);

        // arm_convolve_HWC_q7_fast(conv2_out, CONV3_IM_DIM, CONV3_IM_CH, conv3_w_2, numberOFilters, CONV3_KER_DIM,
		// 				  CONV3_PADDING, CONV3_STRIDE, conv3_b_2, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
		// 				  CONV3_OUT_DIM, conv_buffer, NULL);

        // arm_relu_q7(pool3_out, POOL3_OUT_DIM * POOL3_OUT_DIM * numberOFilters); 
        // arm_maxpool_q7_HWC(conv3_out, POOL3_IM_DIM, POOL3_IM_CH, POOL3_KER_DIM, POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM,  (q7_t *) conv_buffer, pool3_out);
        // arm_relu_q7(pool3_out, POOL3_OUT_DIM * POOL3_OUT_DIM * numberOFilters); 
///////
        
        return 0;
}