
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
#include <stdio.h>
#include <stdlib.h>

int main()
{
	// SETUP SPI
	SPI		   spi(D11, D12, D13); // mosi, miso, sclk
	DigitalOut cs(D0);

    spi.format(16, 3);
    spi.frequency(10000);

	// SPI FORMAT : 4 leading 0 + 12 bits data

	while (1)
	{
		cs = 0;

		// read 16 bits
		uint16_t data = spi.write(0);

		printf("data = %d\n", data);
		ThisThread::sleep_for(chrono::milliseconds(200));


	}

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
