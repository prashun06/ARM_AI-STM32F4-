#include "main.h"
#include "string.h"

#define NUM_FEATURES 2 //n values as input
#define NUM_EXAPLE 3 //total data have 
#define NUM_HID_NOD 3 //number of hid nodes
#define NUM_OUT_NOD 1 //number of out nodes

//note: column for input of a fuct, raw for output of a function
double raw_input[NUM_FEATURES][NUM_EXAPLE] = {{2,5,1}, //data 1
																							{8,5,8}}; //data 2 
	
double raw_output[NUM_OUT_NOD][NUM_EXAPLE] = {200, 90, 190}; //output data 3

double trained_input[NUM_FEATURES][NUM_EXAPLE];
double trained_output[NUM_OUT_NOD][NUM_EXAPLE];

//input to hid weight matrix
double syn0[NUM_HID_NOD][NUM_FEATURES];
//hid to output weight matrix
double syn1[NUM_HID_NOD];

double train_input_eg1[NUM_FEATURES];
double train_output_eg1[NUM_OUT_NOD];
double z1_eg1[NUM_HID_NOD];
double a1_eg1[NUM_HID_NOD];
double z2_eg1;
double yhat_eg1;


int main(){
	uart_tx_init();
	
	normalization_matrix(NUM_FEATURES, NUM_EXAPLE, raw_input,trained_input);
	normalization_matrix(NUM_OUT_NOD, NUM_HID_NOD, raw_output,trained_output);
	
	train_input_eg1[0]=trained_input[0][0];
	train_input_eg1[1]=trained_input[1][0];
	train_output_eg1[0] = trained_output[0][0];
	
	printf("...%f....%f....%f...",train_input_eg1[0],train_input_eg1[1],train_output_eg1[0]);
	
	random_weight_gen_matrix(NUM_HID_NOD,NUM_FEATURES,syn0);
	random_weight_gen_vector(syn1,NUM_OUT_NOD);
	//compute z inside the hidden layer 
	multi_in_multi_out(train_input_eg1,NUM_FEATURES,z1_eg1,NUM_HID_NOD,syn0);
	//compute a (apply sigmoid function in hiddle node)
	vector_sigmoid(z1_eg1,a1_eg1,NUM_FEATURES);
	//compute z2 (prediction_value)
	z2_eg1 = multi_in_sig_out(a1_eg1,syn1,NUM_FEATURES);
	printf("z2 eg1 value : %f...\r\n",z2_eg1);
	//yhat sigmoid for a single value
	yhat_eg1=sigmoid(z2_eg1);
	
	printf("yhat value : %f...\r\n",yhat_eg1);
	while(1){
		 
	}
}
