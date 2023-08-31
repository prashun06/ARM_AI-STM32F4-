
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "usart.h"
#include <stdio.h>
#include <stdint.h>

double sig_in_sig_out_nn(double input, double weight);
double multi_in_sig_out(double * input, double *weight, double INPUT_LEN);
void sin_in_mult_out(double input_scaler,
												double *weight_vector,
													double *output_vector,
														double VECTOR_LEN);

void multi_in_multi_out(double *input_vector,
															uint32_t IN_LEN,
																double *output_vector,
																	uint32_t OUT_LEN,
																		double weighted_matrix[OUT_LEN][IN_LEN]);
void hid_nn(double *input_vector,
							uint32_t IN_LEN,
								uint32_t HID_LEN,
									double in_to_hid_weighted_matrix[HID_LEN][IN_LEN],
										uint32_t OUT_LEN,
											double hid_to_out_weighted_matrix[OUT_LEN][HID_LEN],
												double *output_vector);

double find_err(double yhat, double y);
double find_error(double input, double weight, double expected_value);

void broute_force_learning(double input,
														double weight,
															double expected_value,
																double step_amount,
																	uint16_t iter);

void normalization(double *input_vector, double *output_vector, uint16_t LEN);

void random_weight_gen_matrix(uint32_t HID_LEN, uint32_t INPUT_LEN, double weight_matrix[HID_LEN][INPUT_LEN]);

double sigmoid(double x);
																
void vector_sigmoid(double *input_vector, double *output_vector, uint16_t LEN);

void random_weight_gen_vector(double *weight_vector, uint32_t LEN);

void normalization_matrix(uint32_t ROW, 
														uint32_t COLUMN, 
															double input_matrix[ROW][COLUMN], 	
																double output_matrix[ROW][COLUMN]);
																	//
																