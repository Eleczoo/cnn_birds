
/**
 * @file main.c
 * @author Albanesi, stirnemann
 * @brief MCU firmware implementing birds song detection using tensorflow lite
 * trained model It uses the Nucleo STM32U575ZI board
 * @version 0.1
 * @date 25-04-2024
 *
 *
 * SPI JACK ADC : https://digilent.com/reference/pmod/pmod/mic/ref_manual
 *
 *
 */


#include "mbed.h"

#include <chrono>
#include <ratio>
#include <stdio.h>
#include <stdlib.h>

//
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"
#include <arm_math.h>
// Include cmsis-dsp

// Include Nucleo U575ZI
// #include "stm32u5xx_hal.h"


#define RECORD_TIME	  3					// s
#define SAMPLING_RATE 22050				// Hz
#define SAMPLING_T	  (1000000 / 22050) // us
#define NB_SAMPLES	  (RECORD_TIME * SAMPLING_RATE)

int16_t flip_16_bits(int16_t data);
int16_t bit_extend(int16_t data);
void	print_binary(uint16_t data);
void	callback_audio_record();


Ticker	   audio_ticker;
q31_t	   audio[NB_SAMPLES];
uint32_t   g_index_audio = 0;
SPI		   spi(D11, D12, D13); // mosi, miso, sclk
DigitalOut cs(D0);			   // CHIP SELECCT ADC
bool	   flag_record_finished = false;
bool flag_audio_read = false;

// ! Spectrogram stuff
#define WINDOW_SIZE		  256
#define STEP_SIZE		  128
#define NUMBER_OF_WINDOWS ((NB_SAMPLES - WINDOW_SIZE) / STEP_SIZE + 1)
#define FFT_SIZE		  (WINDOW_SIZE / 2 + 1)


// ! Create the hanning window
q31_t hanning_window_q31[WINDOW_SIZE];
q31_t processed_window_q31[WINDOW_SIZE];
q31_t spectrogram_q31[NUMBER_OF_WINDOWS+10][FFT_SIZE+10];

int main()
{

    printf("WINDOW SIZE : %d\r\n", WINDOW_SIZE);
    printf("NUMBER OF WINDOWS : %d\r\n", NUMBER_OF_WINDOWS);
    printf("FFT SIZE : %d\r\n", FFT_SIZE);
    printf("NB_SAMPLES : %d\r\n", NB_SAMPLES);

	spi.format(16, 3);
	spi.frequency(100000);

	cs = 0;
	spi.write(0);
	cs = 1;
	ThisThread::sleep_for(chrono::milliseconds(200));

	audio_ticker.attach(&callback_audio_record, chrono::microseconds(SAMPLING_T));



	for (size_t i = 0; i < WINDOW_SIZE; i++)
	{
		// calculate the Hanning Window value for i as a float32_t
		float32_t f = 0.5 * (1.0 - arm_cos_f32(2 * PI * i / WINDOW_SIZE));

		// convert value for index i from float32_t to q15_t and
		// store in window at position i
		arm_float_to_q31(&f, &hanning_window_q31[i], 1);

	}
	// ! Create RFFT instance
	arm_rfft_instance_q31 rfft_inst_q31;
	arm_rfft_init_q31(&rfft_inst_q31, WINDOW_SIZE, 0, 1);
    
	while (1)
	{

        if(flag_audio_read)
        {
            flag_audio_read = false;
            cs					   = 0;
            audio[g_index_audio++] = spi.write(0) << 4;
            cs					   = 1;
        }


		if (flag_record_finished)
		{
			flag_record_finished = false;
            audio_ticker.detach();

			printf("BEFORE COMPUTING\r\n");
            
			for (uint32_t i = 0; i < NUMBER_OF_WINDOWS - 1; i++)
			{
                printf("i = %d\r\n", i);

				// ! Apply thehanning window to the audio
				arm_mult_q31(&(audio[i * WINDOW_SIZE]), hanning_window_q31, processed_window_q31,
							 WINDOW_SIZE);

                printf("i after 1 = %d\r\n", i);				// ! Compute the fft

				arm_rfft_q31(&rfft_inst_q31, processed_window_q31, spectrogram_q31[i]);
                
                printf("i after 2 = %d\r\n", i);

				// ! Take the absolute value of the complex numbers
				arm_cmplx_mag_q31(spectrogram_q31[i], spectrogram_q31[i], FFT_SIZE);

                printf("i after 3 = %d\r\n", i);


			}

            printf("FINISHED COMPUTING\r\n");

			// ! Show the spectrogram in the terminal
			for (uint32_t i = 0; i < NUMBER_OF_WINDOWS; i++)
			{
				for (uint32_t j = 0; j < FFT_SIZE; j++)
				{
					printf("%d", spectrogram_q31[i][j]);
				}
				printf("\n");
			}
		}
	}
	return 0;
}


void callback_audio_record()
{

    flag_audio_read = true;

	if (g_index_audio >= NB_SAMPLES)
	{
		g_index_audio		 = 0;
		flag_record_finished = true;
	}
}


// GRAVEYARD
//// Data base format 0000 xxxx xxxx xxxx
//// Where x is the data with lsb first
//// We need to flip the data to get the correct value
// int16_t flip_16_bits(int16_t data)
//{
//	unsigned int nb_bits	  = 16;
//	unsigned int reverse_data = 0;
//	int			 i;
//	for (i = 0; i < nb_bits; i++)
//	{
//		if ((data & (1 << i)))
//			reverse_data |= 1 << ((nb_bits - 1) - i);
//	}
//	return reverse_data;
// }
//// int16_t flip_16_bits(int16_t data)
////{
////	int16_t flipped_data = 0;
////	for (int i = 0; i < 16; i++)
////		flipped_data |= (data & (0x8000 >> i)) >> (15 - i);

////	return flipped_data;
////}


// int16_t bit_extend(int16_t data)
//{
//	uint16_t mask = 0xF000;

//	// int16_t flipped_data = 0;
//	// for (int i = 0; i < 12; i++)
//	//{
//	//	if (data & (1 << i))
//	//	{
//	//		flipped_data |= (1 << (11 - i));
//	//	}
//	// }
//	if (data & 1 << 12) // If it's suppose to be
//	{
//		data |= mask;
//	}
//	return data;
//}


// void print_binary(uint16_t data)
//{
//	for (int i = 0; i < 16; i++)
//	{
//		if (data & (0x8000 >> i))
//			printf("1");
//		else
//			printf("0");
//	}
//	printf("\n");
// }