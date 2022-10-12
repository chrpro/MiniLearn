matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int w = ((l.width - l.size)/l.stride )+ 1;
    int h = ((l.height - l.size) / l.stride )+ 1;
    int c = l.channels;

    matrix out = make_matrix(in.rows, w*h*l.channels);
    // printf("%d \n",in.rows* w*h*l.channels);
    // float val;
   int i, j, k, m, n, b;
   
   for(b = 0; b < in.rows; ++b){
        for (k = 0; k < c; ++k) {
            for (i = 0; i < h; ++i) {
                for (j = 0; j < w; ++j) {
                    // int out_index = j + w*(i + h*(k + c*b));
                    // int out_index = j *b + w*( h*(k + c));
                    int out_index = out.cols*b
                             + w*h*c
                             + w*i + c;
                    printf("%d \n",out_index);
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h =  i*l.stride + n;
                            int cur_w =  j*l.stride + m;
                            int index = cur_w + l.width*(cur_h + l.height*(k + b*l.channels));
                            int valid = (cur_h >= 0 && cur_h < l.height &&
                                         cur_w >= 0 && cur_w < l.width);
                            if (valid == 0)
                            printf("not valid \n");
                            float val = (valid != 0) ? in.data[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    if (out_index > out.rows * out.cols){
                        printf("out.rows * out.cols = %d\n",out.rows * out.cols);
                        printf("index = %d\n",out_index);
                    }
                    if (out_index > l.indexes.rows * l.indexes.cols){
                        printf("**l.indexes.rows * l.indexes.cols= %d\n",l.indexes.rows * l.indexes.cols);
                        printf("**index = %d\n",out_index);
                    }
                    out.data[out_index] = max;
                    l.indexes.data[out_index] = max_i;
                }
            }
        }
    }
    return out;
}
