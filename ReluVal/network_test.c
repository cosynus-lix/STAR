/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 *
 * This is the main file of ReluVal, here is the usage:
 * ./network_test [property] [network] [target] 
 *      [need to print=0] [test for one run=0] [check mode=0]
 *
 * [property]: the saftety property want to verify
 *
 * [network]: the network want to test with
 *
 * [target]: Wanted label of the property
 *
 * [need to print]: whether need to print the detailed info of each split.
 * 0 is not and 1 is yes. Default value is 0.
 *
 * [test for one run]: whether need to estimate the output range without
 * split. 0 is no, 1 is yes. Default value is 0.
 *
 * [check mode]: normal split mode is 0. Check adv mode is 1.
 * Check adv mode will prevent further splits as long as the depth goes
 * upper than 20 so as to locate the concrete adversarial examples faster.
 * Default value is 0.
 * 
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <json.h>
#include <json_util.h>
#include "split.h"

//extern int thread_tot_cnt;

/* print the progress if getting SIGQUIT */
void sig_handler(int signo)
{

    if (signo == SIGQUIT) {
        printf("progress: %d/1024\n", progress);
    }

}



int main( int argc, char *argv[])
{

    //char *FULL_NET_PATH =\
            "nnet/ACASXU_run2a_1_1_batch_2000.nnet";
    char *FULL_NET_PATH;
    char *property_file_name = NULL;

    int target = 0;

    if (argc > 10 || argc < 5) {
        printf("please specify a network\n");
        printf("./network_test [property] [network] "
            "[target] [property_file] [print] "
            "[test for one run] [check mode]\n");
        exit(1);
    }

    for (int i=1; i<argc; i++) {

        if (i == 1) {
            PROPERTY = atoi(argv[i]); 
            if(PROPERTY<0){
                printf("No such property defined");
                exit(1);
            } 
        }

        if (i == 2) {
            FULL_NET_PATH = argv[i];
        }

        if (i == 3) {
            target = atoi(argv[i]);
        }

        if (i == 4) {
            property_file_name = argv[i];
        }

        if (i == 5) {
            NEED_PRINT = atoi(argv[i]);
            if(NEED_PRINT != 0 && NEED_PRINT!=1){
                printf("Wrong print");
                exit(1);
            }
        }

        if (i == 6) {
            NEED_FOR_ONE_RUN = atoi(argv[i]);

            if (NEED_FOR_ONE_RUN != 0 && NEED_FOR_ONE_RUN != 1) {
                printf("Wrong test for one run");
                exit(1);
            }

        }

        if (i == 7) {

            if (atoi(argv[i]) == 0) {
                CHECK_ADV_MODE = 0;
                PARTIAL_MODE = 0;
            }

            if (atoi(argv[i]) == 1) {
                CHECK_ADV_MODE = 1;
                PARTIAL_MODE = 0;
            }

            if (atoi(argv[i]) == 2) {
                CHECK_ADV_MODE = 0;
                PARTIAL_MODE = 1;
            }

        }

    }

    openblas_set_num_threads(1);

    //clock_t start, end;
    srand((unsigned)time(NULL));
    double time_spent;
    int i,j,layer;

    struct NNet* nnet = load_network(FULL_NET_PATH, target);
    
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    
    float input_test[] = {-0.324484, -0.453125,\
                -0.492187, 0.390625, -0.257812};
    //float input_test[] = {59261.709517, -0.046875,\
                1.857353,1145.000000,8.654796};

    struct Matrix input_t = {input_test, 1, 5};

    struct Interval input_interval;
    struct Interval output_to_check_interval;
    
    float u[inputSize], l[inputSize];
    float output_u[outputSize], output_l[outputSize];
    //load_inputs(PROPERTY, inputSize, u, l);

    FILE *f;
    f = fopen(property_file_name, "rb");
    if (! f) {
        printf("Error reading file %s\n", property_file_name);
        exit(1);
    }

    if (load_io(f, inputSize, u, l,
                outputSize, output_u, output_l)) {
        printf("Error reading the input properties");            
    }

    /*for (int i=0;i<inputSize;i++)
        printf("%f ", u[i]);
    printf("\n");
    for (int i=0;i<inputSize;i++)
        printf("%f ", l[i]);
    printf("\n");*/
    


    struct Matrix input_upper = {u,1,nnet->inputSize};
    struct Matrix input_lower = {l,1,nnet->inputSize};
    

    input_interval.lower_matrix = input_lower;
    input_interval.upper_matrix = input_upper;

    struct Matrix output_upper = {output_u,nnet->outputSize,1};
    struct Matrix output_lower = {output_l,nnet->outputSize,1};
    output_to_check_interval.lower_matrix = output_lower;
    output_to_check_interval.upper_matrix = output_upper;
     
    float grad_upper[inputSize], grad_lower[inputSize];
    struct Interval grad_interval = {
                (struct Matrix){grad_upper, 1, inputSize},
                (struct Matrix){grad_lower, 1, inputSize}
            };

    //normalize_input(nnet, &input_t);
    normalize_input_interval(nnet, &input_interval);

    float o[nnet->outputSize];
    struct Matrix output = {o, outputSize, 1};

    float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
    struct Interval output_interval = {
                (struct Matrix){o_lower, outputSize, 1},
                (struct Matrix){o_upper, outputSize, 1}
            };

    //if (signal(SIGQUIT, sig_handler) == SIG_ERR)
        //printf("\ncan't catch SIGQUIT\n");

    int n = 0;
    int feature_range_length = 0;
    int split_feature = -1;
    int depth = 0;

    /*
    printf("running property %d with network %s\n",\
                PROPERTY, FULL_NET_PATH);
    printf("input ranges:\n");

    printMatrix(&input_interval.upper_matrix);
    printMatrix(&input_interval.lower_matrix);
    
    printf("safe output:\n");

    printMatrix(&output_to_check_interval.upper_matrix);
    printMatrix(&output_to_check_interval.lower_matrix);
    */

    for (int i=0;i<inputSize;i++) {

        if (input_interval.upper_matrix.data[i] <\
                input_interval.lower_matrix.data[i]) {
            printf("wrong input!\n");
            exit(0);
        }

        if(input_interval.upper_matrix.data[i] !=\
                input_interval.lower_matrix.data[i]){
            n++;
        }

    }

    feature_range_length = n;
    int *feature_range = (int*)malloc(n*sizeof(int));

    for (int i=0, n=0;i<nnet->inputSize;i++) {
        if(input_interval.upper_matrix.data[i] !=\
                input_interval.lower_matrix.data[i]){
            feature_range[n] = i;
            n++;
        }
    }

    struct Interval* initial_input = copy_interval(&input_interval);
   
    //evaluate(nnet, &input_t, &output);
    //forward_prop(nnet, &input_t,&output);
    //printMatrix(&output);
    
    gettimeofday(&start, NULL);
    int isOverlap = 0;
    float avg[100] = {0};
     
    PartitionInput partition_input = {
        nnet,
        &input_interval,
        &output_to_check_interval,
        feature_range, feature_range_length, split_feature,
        0.8,0.001,
        initial_input
    };

    PartitionList *partitions = compute_partitioning(&partition_input);

    gettimeofday(&finish, NULL);
    time_spent = ((float)(finish.tv_sec - start.tv_sec) *\
            1000000 + (float)(finish.tv_usec - start.tv_usec)) /\
            1000000;

    /*if (isOverlap == 0 && adv_found == 0) {
        printf("\nNo adv!\n");
    }

    
    printf("time: %f \n\n\n", time_spent);*/
    
    // Create json file for storing splits
    
    const char *filename = "./splits.json";
    json_object * jobj = json_object_new_object();

    json_object * reach = json_object_new_object();
    json_object *reach_lower = json_object_new_array();
    json_object *reach_upper = json_object_new_array();
    
    json_object * no_reach = json_object_new_object();
    json_object *no_reach_lower = json_object_new_array();
    json_object *no_reach_upper = json_object_new_array();
    
    for (int p = 0; p < partitions->safe_size; p++) {    
        json_object *ru = json_object_new_array();
        json_object *rl = json_object_new_array();
        for (int i = 0; i < inputSize; i++) {
            char val[30] = {0};
            sprintf(val, "%.5f", partitions->safe_partitions[p].upper_matrix.data[i]);
            json_object_array_add(ru,json_object_new_string(val)); 
            sprintf(val, "%.5f", partitions->safe_partitions[p].lower_matrix.data[i]);  
            json_object_array_add(rl,json_object_new_string(val));

            }
        json_object_array_add(reach_upper,ru);
        json_object_array_add(reach_lower,rl);
        //printMatrix(&partitions->safe_partitions[p].upper_matrix);
        //printMatrix(&partitions->safe_partitions[p].lower_matrix);
    }
    
    for (int p = 0; p < partitions->unsafe_size; p++) {  
        json_object *nru = json_object_new_array();
        json_object *nrl = json_object_new_array();
        for (i = 0; i < inputSize; i++) {
            char val[30] = {0};
            sprintf(val, "%.5f", partitions->unsafe_partitions[p].upper_matrix.data[i]);
            json_object_array_add(nru,json_object_new_string(val)); 
            sprintf(val, "%.5f", partitions->unsafe_partitions[p].lower_matrix.data[i]);  
            json_object_array_add(nrl,json_object_new_string(val));
            }
        json_object_array_add(no_reach_upper,nru);
        json_object_array_add(no_reach_lower,nrl);
    }
    
    json_object_object_add(reach,"upper", reach_upper);
    json_object_object_add(reach,"lower", reach_lower);
    json_object_object_add(jobj,"reach", reach);
    json_object_object_add(no_reach,"upper", no_reach_upper);
    json_object_object_add(no_reach,"lower", no_reach_lower);
    json_object_object_add(jobj,"no_reach", no_reach);

    //printf ("The json object created: %sn",json_object_to_json_string(jobj));
    if (json_object_to_file(filename, jobj))
      printf("Error: failed to save %s!!\n", filename);
    else
      printf("%s saved.\n", filename);

    destroy_network(nnet);
    free(feature_range);

}
