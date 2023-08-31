#include <stdio.h>
#include <stdint.h>
//#include "sin_in_sin_out.h"

extern void UART_Write_String(char *p);
extern void UART_read_string(char *p_read, uint16_t num_bytes);
extern void uart_tx_init(void);

extern char uart_read(void);
extern void test_setup(void);

extern double sig_in_sig_out_nn(double input, double weight);
extern double multi_in_sig_out(double * input, double *weight, double INPUT_LEN);
extern void sin_in_mult_out(double input_scaler,
												double *weight_vector,
													double *output_vector,
														double VECTOR_LEN);

extern void multi_in_multi_out(double *input_vector,
																	uint32_t IN_LEN,
																		double *output_vector,
																			uint32_t OUT_LEN,
																				double weighted_matrix[OUT_LEN][IN_LEN]);
				
extern void hid_nn(double *input_vector,
											uint32_t IN_LEN,
												uint32_t HID_LEN,
													double in_to_hid_weighted_matrix[HID_LEN][IN_LEN],
														uint32_t OUT_LEN,
															double hid_to_out_weighted_matrix[OUT_LEN][HID_LEN],
																double *output_vector);																

extern double find_err(double yhat, double y);
extern double find_error(double input, double weight, double expected_value);				

extern void broute_force_learning(double input,
														double weight,
															double expected_value,
																double step_amount,
																	uint16_t iter);															

extern void normalization(double *input_vector, double *output_vector, uint16_t LEN);
extern void random_weight_gen_matrix(uint32_t HID_LEN, uint32_t INPUT_LEN, double weight_matrix[HID_LEN][INPUT_LEN]);
extern void vector_sigmoid(double *input_vector, double *output_vector, uint16_t LEN);
extern void random_weight_gen_vector(double *weight_vector, uint32_t LEN);
extern void normalization_matrix(uint32_t ROW, 
																	uint32_t COLUMN, 
																		double input_matrix[ROW][COLUMN], 	
																			double output_matrix[ROW][COLUMN]);
													
extern double sigmoid(double x);
																		//


																