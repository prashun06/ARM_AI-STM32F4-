
#include "sin_in_sin_out.h"


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

	return powf(((input*weight)-expected_value),2);  //for all the iter
}

void broute_force_learning(double input,
														double weight,
															double expected_value,
																double step_amount,
																	uint16_t iter){
	double prediction, error;
	double up_prediction, down_predicton, up_error, down_error;
	
	for (int i=0; i<iter;i++){
		prediction=input*weight;
		error= powf(((input*weight)-expected_value),2);
		
		printf("Error : %f  prediction :%f ....\r\n",error,prediction);
		
		up_prediction = input*(weight+step_amount);
		up_error = powf((up_prediction-expected_value),2);
		down_predicton = input*(weight-step_amount);
		down_error = powf((down_predicton-expected_value),2);
		
		if(down_error<up_error){
			weight = weight-step_amount;
		}
		if(down_error>up_error){
			weight = weight+step_amount;
		}
	}
}
																	
void normalization(double *input_vector, double *output_vector, uint16_t LEN){
	//find the max 
	double max=input_vector[0];
	for(int i=0;i<LEN; i++){
		if(input_vector[i]>max){
			max = input_vector[i];
		}
	}
	
	for(int i=0;i<LEN; i++){
		output_vector[i]=input_vector[i]/max;
	}
}

void random_weight_gen_matrix(uint32_t HID_LEN, uint32_t INPUT_LEN, double weight_matrix[HID_LEN][INPUT_LEN]){

	double d_rand;
	//seed random number generator 
	srand(1);
	//generate random number for the weight 
	for(int i=0; i<HID_LEN;i++){
		for(int j=0;j<INPUT_LEN;j++){
			d_rand = (rand()%10);
			d_rand =d_rand/10;
			
			weight_matrix[i][j]=d_rand;
		}
	}
}

double sigmoid(double x){

	return  1/(1+exp(-x));
}

void vector_sigmoid(double *input_vector, double *output_vector, uint16_t LEN){
	for(int i=0;i<LEN;i++){
		output_vector[i]=sigmoid(input_vector[i]);
	}
}

void normalization_matrix(uint32_t ROW, 
														uint32_t COLUMN, 
															double input_matrix[ROW][COLUMN], 	
																double output_matrix[ROW][COLUMN]){
	//find the max 
	double max=-99999999;
	for(int i=0;i<ROW; i++){
		for(int j=0;j<COLUMN;j++)
			if(input_matrix[i][j]>max){
				max = input_matrix[i][j]; //take the max value of whole matrix
		}
	}
	
	for(int i=0;i<ROW; i++){
		for(int j=0;j<COLUMN;j++){
			output_matrix[i][j]=input_matrix[i][j]/max;
		}	
	}
}

void random_weight_gen_vector(double *weight_vector, uint32_t LEN){

	double d_rand;
	//seed random number generator 
	srand(1);
	//generate random number for the weight 
	for(int i=0; i<LEN;i++){
			d_rand = (rand()%10);
			d_rand =d_rand/10;
			
			weight_vector[i]=d_rand;
	}
}

