#include "main.h"
#include "string.h"

#define SAD_IND     0
#define SICK_IND    1
#define ACTIVE_IND  2

#define OUT_IND   3
#define IN_IND    3
#define HID_IND   3

double predicted_output[OUT_IND];
																						//temp  hum  airq
double in_to_hid_weights[HID_IND][IN_IND]={{-2.0, 9.5, 2.01},  //hid1
																						{-0.8, 7.2, 6.3},  //hid2
																						{-0.5, 0.45, 0.9}}; //hid3

																				
																					//hid1  hid2   hid3
double hid_to_out_weight[OUT_IND][HID_IND]={{-1.0,  1.15,   0.11},  //sad
																						{-0.18,  0.15, -0.01}, //sick
																						{0.25, -0.25, -0.1}}; //active


double input[IN_IND]={30.0, 87.0, 110.0};		

double expected_value[OUT_IND] = {600,10,-90}; //y values

int main(){
	uart_tx_init();
	
	hid_nn(input,IN_IND,HID_IND,in_to_hid_weights,OUT_IND,hid_to_out_weight,predicted_output);
	printf("pri temp output: %f...\r\n",predicted_output[SAD_IND]);
	printf("error temp: %f...\r\n",find_err(predicted_output[SAD_IND], expected_value[SAD_IND]));
	
	printf("pri hum output: %f...\r\n",predicted_output[SICK_IND]);
	printf("error hum: %f...\r\n",find_err(predicted_output[SICK_IND], expected_value[SICK_IND]));
	
	printf("pri airq output: %f...\r\n",predicted_output[ACTIVE_IND]);
	printf("error airq: %f...\r\n",find_err(predicted_output[ACTIVE_IND], expected_value[ACTIVE_IND]));
	while(1){
		 
	}
}
