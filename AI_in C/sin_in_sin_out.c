
#include "sin_in_sin_out.h"
#include <math.h>

double sig_in_sig_out_nn(double input, double weight){
	return (input*weight); 
}

double weighted_sum(double * input, double *weight, double INPUT_LEN){

	double output;
	for(int i=0; i<INPUT_LEN;i++){
		output+=input[i]*weight[i];
	}
	return output;
}

double multi_in_sig_out(double * input, double *weight, double INPUT_LEN){

	double predicted_value = weighted_sum(input,weight,INPUT_LEN);
	return predicted_value;
}

void element_mult(double input_scaler,
									double *weight_vector,
										double *output_vector,
											double VECTOR_LEN){
		
	for(int i=0;i<VECTOR_LEN;i++){	
			output_vector[i] = input_scaler * weight_vector[i];
	}
}

void sin_in_mult_out(double input_scaler,
												double *weight_vector,
													double *output_vector,
														double VECTOR_LEN){
		
	element_mult(input_scaler,weight_vector,output_vector,VECTOR_LEN);										
}

void matrix_vector_multiple(double *input_vector,
															uint32_t IN_LEN,
																double *output_vector,
																	uint32_t OUT_LEN,
																		double weighted_matrix[OUT_LEN][IN_LEN]){

		for(int k=0;k<OUT_LEN;k++){
			for (int j=0;j<IN_LEN;j++){
				output_vector[k] += input_vector[j]*weighted_matrix[k][j];
			}
		}
}
																		
void multi_in_multi_out(double *input_vector,
															uint32_t IN_LEN,
																double *output_vector,
																	uint32_t OUT_LEN,
																		double weighted_matrix[OUT_LEN][IN_LEN]){
																		
	matrix_vector_multiple(input_vector,IN_LEN,output_vector,OUT_LEN,weighted_matrix);
}
																		
void hid_nn(double *input_vector,
							uint32_t IN_LEN,
								uint32_t HID_LEN,
									double in_to_hid_weighted_matrix[HID_LEN][IN_LEN],
										uint32_t OUT_LEN,
											double hid_to_out_weighted_matrix[OUT_LEN][HID_LEN],
												double *output_vector){
	
	double hid_pred_values[HID_LEN];
	multi_in_multi_out(input_vector,IN_LEN,hid_pred_values,HID_LEN,in_to_hid_weighted_matrix);
	multi_in_multi_out(hid_pred_values,HID_LEN,output_vector,OUT_LEN,hid_to_out_weighted_matrix);
													
}

double find_err(double yhat, double y){
	return powf((yhat-y),2);  //error2
}

double find_error(double input, double weight, double expected_value){

	return powf(((input*weight)-expected_value),2);
}

