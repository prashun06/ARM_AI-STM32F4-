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
//


																