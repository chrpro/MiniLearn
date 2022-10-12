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

// #include "train_data.h"
// Test data UCI har
// #include "test_data.h"
//Train data UCI har

#include "keyspot_test.h"
#include "keyspot_train_data.h"



// Neural Network
#include "nnom.h"
#include "kws_weights.h"
#include "cmsis-w.h"



#define CLASSES 5

#define numberOFilters 8





#define CONV3_OUT_Q 4

#define MAX_CONV_BUFFER_SIZE 1576

static q7_t conv3_out[numberOFilters*9*2];

static q7_t conv3_w[CONV3_WT_SHAPE] =  CONV3_WT;
static q7_t conv3_b[CONV3_BIAS_SHAPE] = CONV3_BIAS;

static q7_t conv3_w_2[numberOFilters*64*3*3] =  {0};
static q7_t conv3_b_2[numberOFilters] = {0};


static q15_t conv_buffer[MAX_CONV_BUFFER_SIZE];

nnom_status_t callback(nnom_layer_t* layer)
{
	float scale = 1 << (layer->out->tensor->q_dec[0]);
	printf("\nOutput of Layer %s", default_layer_names[layer->type]);
	for (int i = 0; i < (int)tensor_size(layer->out->tensor); i++)
	{
		if (i % 16 == 0)
			printf("\n");
		printf("%f ", (float)((int8_t*)layer->out->tensor->p_data)[i] / scale);
	}
	printf("\n");
	return NN_SUCCESS;
}



void insert(int item, float itprio);
void del();
void display();

struct node
{
    float priority;
    int info;
    struct node *next;
} *start = NULL, *q, *temp, *new;

typedef struct node N;


void insert( int item, float itprio)
{
    // int item, itprio; 
    new = (N *)malloc(sizeof(N));

    new->info = item;
    new->priority = itprio;
    new->next = NULL;
    if (start == NULL)
    {
        start = new;
    }
    else if (start != NULL && itprio < start->priority)
    {
        new->next = start;
        start = new;
    }
    else
    {
        q = start;
        while (q->next != NULL && q->next->priority < itprio)
        {
            q = q->next;
        }
        new->next = q->next;
        q->next = new;
    }
}





void fc_2l_network_init(net * fully_con, int input_f_layer, int out_f_layer );

int main(void)
{
    //initialize the random gener  //seed with time in microseconds (works with Unix systems)
    struct timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    // srand(1);

    //fully connected size & parameters
    //------------Learning Parameters--------------
    int fc_layer_out = 96;
    int batch = 4;
    // int iterations = 35;
    float rate = .0001;
    float momentum = .9;
    float decay = .0001;
    //-----------------------------------------------
    const char chalmersBanner[] = {
		"============================================================\n"
		"Tranfers Learning on low-power IoT devices                  \n"
		"Chalmers University                                         \n"
		"============================================================\n"
	}; 
    //print bannger to indicate starting...
    printf(chalmersBanner);
    uint32_t index = 0;
    int CNN_out_Size = 0;//tensor_size(layer->out->tensor);

    nnom_model_t* model;
	nnom_predict_t * pre;
	int8_t* input;
	float prob;
	uint32_t label;
	size_t size = 0;
    //---------------------------*****SET UP ****----------------
	model = nnom_model_create();// create NNoM model , parameters in weigth.h , backend CMSIS
    nnom_layer_t *cnn_layer;
    cnn_layer = model->head;

    uint32_t run_num = 0;
    while (run_num < 6){
        run_num++;
        if(run_num == 6){
            CNN_out_Size = tensor_size(cnn_layer->out->tensor);
            // nnom_status_t p = callback( rnn_layer);
            printf("CNN last layer output size %d\n",CNN_out_Size);
            // printf("Num of filters %d\n",cnn_layer->filter_mult);

            // NNOM_LOG("%s/", default_layer_names[layer->type]);
            // NNOM_LOG("%s\n ", default_cell_names[((nnom_rnn_layer_t*)layer)->cell->type]);
            // print_variable(((nnom_rnn_layer_t*)layer)->cell->out_data, "q7 output)", 7, ((nnom_rnn_layer_t*)layer)->cell->units);
            // print_variable(cell->out_data, "q7 output)", 7, cell->units);
            // NNOM_LOG("%f\n ", (float)[0] );
            break;
        }
        
        if (cnn_layer->shortcut == NULL)
            break;

        cnn_layer = cnn_layer->shortcut;
    }


    //------------------------------------------------------
    //*** Example of model run and prediction
	pre = prediction_create(model, nnom_output_data, sizeof(nnom_output_data), 3); // network, Number of classes, get top-4
    memcpy(nnom_input_data, TEST_IMAGES[0], sizeof(TEST_IMAGES[0]));

    prediction_run(pre, TEST_LABELS[0] , &label, &prob);  // this provide more info but requires prediction API
    //****
    printf("label %d, yHat = %f\n", label, prob);

    //------------------------------------------------------

    //-----------------Transfer Learning Part-------------------
    // init networks
    // net transfer_learning = {0}; 
    // fc_2l_network_init(&transfer_learning, RNN_out_Size, fc_layer_out);
    // init data]
    int sub_sample = 80;
    data training_data;
    matrix q_X0 = make_matrix(sub_sample,CNN_out_Size);
    matrix q_Y0 = make_matrix(sub_sample,CLASSES);

 
	float scale = 1 << (cnn_layer->out->tensor->q_dec[0]);



	// for (int i = 0; i < (int)tensor_size(rnn_layer->out->tensor); i++)
	// {

	// 	(float)((int8_t*)rnn_layer->out->tensor->p_data)[i] / scale ;
	// }

    printf("CNN OUT size %d\n\n",CNN_out_Size);
    for(int imgi=0; imgi < sub_sample; imgi++)
    {
        memcpy(nnom_input_data, NODE_0_TRAIN_IMAGES[imgi], sizeof(nnom_input_data));
        prediction_run(pre, NODE_0_TRAIN_LABELS[imgi] , &label, &prob);  // this provide more info but requires prediction API
        // index = network(NODE_0_TRAIN_IMAGES[imgi]);
        for (int i = 0; i < CNN_out_Size; i++){
            q_X0.data[ imgi*q_X0.cols + i] = (float)((int8_t*)cnn_layer->out->tensor->p_data)[i] / scale ;
            // q_X0.data[ imgi*q_X0.cols + i] = (float) pool3_out[i] ;/// (float) pow(2,CONV3_OUT_Q) ;
            // q_X0.data[ imgi*q_X0.cols + i] = (float)((int8_t*)layer->out->tensor->p_data)[i] / scale;
            //  (float)conv2_out[i] ;/// (float) pow(2,CONV3_OUT_Q) ;
            //matrix.type(torch.float32)/2**q
            // quantized = weight * (2 ** q_frac)
            // dequantize = out / 2**q
        
        }
        q_Y0.data[ (imgi*q_Y0.cols)+ NODE_0_TRAIN_LABELS[imgi] ] = 1;     
    }
    training_data.x= q_X0;
    training_data.y = q_Y0;

    // matrix X_test = make_matrix(TOTAL_TEST_IMAGES,CNN_out_Size);
    // matrix Y_test = make_matrix(TOTAL_TEST_IMAGES,CLASSES);

    // for(int imgi=0; imgi < TOTAL_TEST_IMAGES; imgi++)
    // {
    //     memcpy(nnom_input_data, TEST_IMAGES[imgi], sizeof(nnom_input_data));
    //     prediction_run(pre, TEST_LABELS[imgi] , &label, &prob);  // this provide more info but requires prediction API
    //     // index = network(TEST_IMAGES[imgi]);
    //     for (int i = 0; i < CNN_out_Size; i++){
    //         X_test.data[ imgi*X_test.cols + i] = (float)((int8_t*)cnn_layer->out->tensor->p_data)[i] / scale ;
    //         // (float)conv2_out[i];/// (float) pow(2,CONV3_OUT_Q) ;
    //         // X_test.data[ imgi*X_test.cols + i] = pool3_out[i] ;// / (float) pow(2,CONV3_OUT_Q) ;
    //     }
    //     Y_test.data[ (imgi*Y_test.cols)+ TEST_LABELS[imgi] ] = 1;      
    // }

    // data test;
    // test.x=X_test;
    // test.y=Y_test;



    // train_image_classifier(transfer_learning, training_data, batch, iterations,  rate, momentum, decay);
    // printf("=> Tranfers learning Training accuracy: %f\n", accuracy_net(transfer_learning, training_data));
    // printf("=> Tranfers learning Testing  accuracy: %f\n", accuracy_net(transfer_learning, test));
    

    
    // int iterations = NODE_0_TOTAL_TRAIN_IMAGES / batch;
    // int iterations = 80;
    // printf("iterations = %d\n",iterations);
    

    static net active_learning = {0}; 
    active_learning.n = 4;
    active_learning.layers = calloc(4, sizeof(layer));   

    // fully_con->layers[0] = make_convolutional_layer(11, 4, 64, numberOFilters, 3 , 1);
    active_learning.layers[0] = make_convolutional_layer(11, 4, 64, numberOFilters, 3 , 1);
    
    int filterSize =  64*3*3;

    for (int index = 0; index < filterSize * numberOFilters; index++)
    {
        active_learning.layers[0].w.data[index] = (float) (conv3_w[index]  ) /  powf(2, CONV3_WEIGHT_Q); 
    }

    for (int index = 0; index < numberOFilters; index++)
    {
        active_learning.layers[0].b.data[index] = (float)  conv3_b[index] /  powf(2, CONV3_BIAS_Q);
    }
    active_learning.layers[1] = make_activation_layer(RELU);
    // active_learning.layers[2] = make_maxpool_layer( POOL3_IM_DIM, POOL3_IM_DIM, POOL3_IM_CH,  POOL3_KER_DIM , POOL3_STRIDE); 
    // active_learning.layers[2] = make_maxpool_layer( POOL3_IM_DIM, POOL3_IM_DIM, numberOFilters,  POOL3_KER_DIM , POOL3_STRIDE); 
    
    // int convout = numberOFilters*POOL3_OUT_DIM*POOL3_OUT_DIM;
    int convout = numberOFilters*9*2;
    active_learning.layers[2] = make_connected_layer(convout, CLASSES); 
    active_learning.layers[2].freeze = 0;
    active_learning.layers[3] = make_activation_layer(SOFTMAX);

        
    int iterations = 80;

    for ( int epoch = 1; epoch < 7; epoch++){
        train_image_classifier(active_learning, training_data, batch, iterations,  rate, momentum, decay);
        
        float train_acc = accuracy_net(active_learning, training_data);
        printf("%d :: train acc = %f & \n",epoch, train_acc);
        // float test_acc = accuracy_net(active_learning, test);
        // printf("test acc = %f\n", test_acc);

    }

    printf("\n-----------------------------------------------------------------------------\n");

    for (int index = 0; index < filterSize * numberOFilters; index++){
        conv3_w_2[index] = (int) ceilf (active_learning.layers[0].w.data[index] *  powf(2, CONV3_WEIGHT_Q) );
    }
    for (int index = 0; index < numberOFilters; index++){
        conv3_b_2[index] = (int) ceilf ( active_learning.layers[0].b.data[index] * powf(2, CONV3_BIAS_Q) );
    }


    int convOut = numberOFilters*9*2;


    active_learning.n = 4;
    active_learning.layers = calloc(4, sizeof(layer));   
    active_learning.layers[0] = make_connected_layer(convOut, 92);
    active_learning.layers[0].freeze =0;
    active_learning.layers[1] =  make_activation_layer(RELU);

    active_learning.layers[2] = make_connected_layer(92 , CLASSES);
    active_learning.layers[2].freeze =0;

    active_learning.layers[3] = make_activation_layer(SOFTMAX);

    data training_data2;
    matrix q_X = make_matrix(NODE_0_TOTAL_TRAIN_IMAGES,convOut);
    matrix q_Y = make_matrix(NODE_0_TOTAL_TRAIN_IMAGES,CLASSES);

    // float scale = 1 << CONV1_OUT_Q;
    for(int img_row=0; img_row < NODE_0_TOTAL_TRAIN_IMAGES; img_row++)
    {
        memcpy(nnom_input_data, NODE_0_TRAIN_IMAGES[img_row], sizeof(nnom_input_data));
        prediction_run(pre, NODE_0_TRAIN_LABELS[img_row] , &label, &prob);  // this provide more info but requires prediction API
        
        
        q7_t convp_out[CNN_out_Size];


        for (int i = 0; i < CNN_out_Size; i++){
           convp_out[i] = (q7_t) ((int8_t*)cnn_layer->out->tensor->p_data)[i];      
        }

        arm_convolve_HWC_q7_basic_nonsquare(convp_out, 11, 4, 64,  conv3_w_2, numberOFilters, CONV3_KER_DIM,CONV3_KER_DIM,
						  CONV3_PADDING,CONV3_PADDING, CONV3_STRIDE,CONV3_STRIDE, conv3_b_2, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
						  9,2, conv_buffer, NULL);



        for (int i = 0; i < convOut; i++){
            q_X.data[ img_row * q_X.cols + i] = (float) conv3_out[i] /  powf(2,CONV3_OUT_Q) ;        
        }


        q_Y.data[ (img_row * q_Y.cols) + NODE_0_TRAIN_LABELS[img_row] ] = 1;     
    }
    
    training_data2.x= q_X;
    training_data2.y = q_Y;



    matrix X_test2 = make_matrix(TOTAL_TEST_IMAGES,convOut);
    matrix Y_test2 = make_matrix(TOTAL_TEST_IMAGES,CLASSES);


    // for(int imgi=0; imgi < TOTAL_TEST_IMAGES; imgi++)
    // {
    //     memcpy(nnom_input_data, TEST_IMAGES[imgi], sizeof(nnom_input_data));
    //     prediction_run(pre, TEST_LABELS[imgi] , &label, &prob);  // this provide more info but requires prediction API
    //     // index = network(TEST_IMAGES[imgi]);
    //     for (int i = 0; i < CNN_out_Size; i++){
    //         X_test.data[ imgi*X_test.cols + i] = (float)((int8_t*)cnn_layer->out->tensor->p_data)[i] / scale ;
    //         // (float)conv2_out[i];/// (float) pow(2,CONV3_OUT_Q) ;
    //         // X_test.data[ imgi*X_test.cols + i] = pool3_out[i] ;// / (float) pow(2,CONV3_OUT_Q) ;
    //     }
    //     Y_test.data[ (imgi*Y_test.cols)+ TEST_LABELS[imgi] ] = 1;      
    // }

    // data test;
    // test.x=X_test;
    // test.y=Y_test;



   for(int imgi=0; imgi < TOTAL_TEST_IMAGES; imgi++)
    {
        memcpy(nnom_input_data, TEST_IMAGES[imgi], sizeof(nnom_input_data));

        prediction_run(pre, TEST_LABELS[imgi] , &label, &prob);  // this provide more info but requires prediction API\
        q7_t convp_out[CNN_out_Size];
        
        q7_t convp_out[CNN_out_Size];
        for (int i = 0; i < CNN_out_Size; i++){
           convp_out[i] = (q7_t) ((int8_t*)cnn_layer->out->tensor->p_data)[i];      
        }

        arm_convolve_HWC_q7_basic_nonsquare(convp_out, 11, 4, 64,  conv3_w_2, numberOFilters, CONV3_KER_DIM,CONV3_KER_DIM,
						  CONV3_PADDING,CONV3_PADDING, CONV3_STRIDE,CONV3_STRIDE, conv3_b_2, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
						  9,2, conv_buffer, NULL);

        for (int i = 0; i < convOut; i++){
            X_test2.data[ imgi*X_test2.cols + i] = (float) conv3_out[i] /  powf(2,CONV3_OUT_Q) ;
        }
        Y_test2.data[ (imgi*Y_test2.cols) + TEST_LABELS[imgi] ] = 1;      
    }

    data test2;
    test2.x=X_test2;
    test2.y=Y_test2;




    for ( int epoch = 1; epoch < 35; epoch++){
        train_image_classifier(active_learning, training_data2, batch, iterations,  rate, momentum, decay);
        float train_acc = accuracy_net(active_learning, training_data2);
        printf("%d :: train acc = %f & ",epoch, train_acc);
        float test_acc = accuracy_net(active_learning, test2);
        printf("test acc = %f\n", test_acc);

    }

    float acc2 = accuracy_net(active_learning, test2);
    FILE* fp ;
    fp = fopen("Acc8-5classes-600-samples.csv", "a"); // save result in csv file
    fprintf(fp, "%f, \n", acc2);
    fclose(fp);





	prediction_delete(pre);
	// model_stat(model);
	model_delete(model);
    return 0;


}
    // printf("%s\n",filename);
    // FILE *fp = fopen(filename, "w");  // open for reading

    // fprintf(fp, "Train,Test\n");	

    // net transfer_learning = {0}; 
    // fc_2l_network_init(&transfer_learning, CNN_out_Size, fc_layer_out);
    // for ( int epoch = 1; epoch < 15 ; epoch++){

    //     train_image_classifier(transfer_learning, training_data, batch, iterations,  rate, momentum, decay);
    //     float train_acc = accuracy_net(transfer_learning, training_data);
    //     printf("Epoch:%d=> TF Training accuracy: %f",epoch,train_acc);
    //     float test_acc = accuracy_net(transfer_learning, test);
    //     printf("Epoch:%d=> TF Test  accuracy: %f\n\n",epoch, test_acc);
    //                 // save results
    //     // fprintf(fp, "%f, %f \n", test_acc,train_acc);
    //     // fprintf(fp, "%f,%f\n", train_acc, test_acc);
    // // free_connected_layer(transfer_learning.layers[0]);

    // }
    // fclose(fp);
    // float test_acc = accuracy_net(transfer_learning, test);
    // printf("Test  accuracy: %f\n\n", test_acc);
    // free(&transfer_learning.layers);


    // print prediction result
	// prediction_end(pre);
	// prediction_summary(pre);
	// prediction_delete(pre);
	// // model_stat(model);
	// model_delete(model);


    // return 0;




// void fc_2l_network_init(net * fully_con, int input_f_layer, int out_f_layer ){
//     fully_con->n = 5;
//     fully_con->layers = calloc(5, sizeof(layer));

//     fully_con->layers[0] = make_convolutional_layer(11, 4, 64, numberOFilters, 3 , 1);
    


//     fully_con->layers[1] = make_connected_layer(576, out_f_layer);
//     fully_con->layers[1].freeze = 0 ;
    
//     fully_con->layers[2] =  make_activation_layer(RELU);

//     fully_con->layers[3] = make_connected_layer(out_f_layer , CLASSES);
//     fully_con->layers[3].freeze = 0 ;

//     fully_con->layers[4] = make_activation_layer(SOFTMAX);
// }
