
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
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"


#define RECORD_TIME	  3					// s
#define SAMPLING_RATE 22050				// Hz
#define SAMPLING_T	  (1000000 / 22050) // us
#define NB_SAMPLES	  (RECORD_TIME * SAMPLING_RATE)

int16_t flip_16_bits(int16_t data);
int16_t bit_extend(int16_t data);
void	print_binary(uint16_t data);
void	callback_audio_record();


Ticker	   flipper;
uint16_t   audio[NB_SAMPLES];
uint32_t   g_index_audio = 0;
SPI		   spi(D11, D12, D13); // mosi, miso, sclk
DigitalOut cs(D0);			   // CHIP SELECCT ADC


int main()
{
	spi.format(16, 3);
	spi.frequency(100000);

	cs = 0;
	spi.write(0);
	cs = 1;
	ThisThread::sleep_for(chrono::milliseconds(200));


	flipper.attach(&callback_audio_record, chrono::microseconds(SAMPLING_T));

	while (1)
	{
		// Read audio data from microphone

		// Run the model

		// Check if the model detected a bird song

		// If a bird song is detected, turn on the builtin led

		// If a bird song is not detected, turn off the builtin led
	}
	return 0;
}


void callback_audio_record()
{
	cs									= 0;
	audio[g_index_audio++ % NB_SAMPLES] = spi.write(0) << 4;
	cs									= 1;
}

// Data base format 0000 xxxx xxxx xxxx
// Where x is the data with lsb first
// We need to flip the data to get the correct value
int16_t flip_16_bits(int16_t data)
{
	unsigned int nb_bits	  = 16;
	unsigned int reverse_data = 0;
	int			 i;
	for (i = 0; i < nb_bits; i++)
	{
		if ((data & (1 << i)))
			reverse_data |= 1 << ((nb_bits - 1) - i);
	}
	return reverse_data;
}
// int16_t flip_16_bits(int16_t data)
//{
//	int16_t flipped_data = 0;
//	for (int i = 0; i < 16; i++)
//		flipped_data |= (data & (0x8000 >> i)) >> (15 - i);

//	return flipped_data;
//}


int16_t bit_extend(int16_t data)
{
	uint16_t mask = 0xF000;

	// int16_t flipped_data = 0;
	// for (int i = 0; i < 12; i++)
	//{
	//	if (data & (1 << i))
	//	{
	//		flipped_data |= (1 << (11 - i));
	//	}
	// }
	if (data & 1 << 12) // If it's suppose to be
	{
		data |= mask;
	}
	return data;
}


void print_binary(uint16_t data)
{
	for (int i = 0; i < 16; i++)
	{
		if (data & (0x8000 >> i))
			printf("1");
		else
			printf("0");
	}
	printf("\n");
}