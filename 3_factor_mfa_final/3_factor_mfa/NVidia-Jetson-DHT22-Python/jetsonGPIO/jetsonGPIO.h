/*
 * jetsonGPIO.h
 *
 * Copyright (c) 2015 JetsonHacks
 * www.jetsonhacks.com
 *
 * Based on Software by RidgeRun
 * Originally from:
 * https://developer.ridgerun.com/wiki/index.php/Gpio-int-test.c
 * Copyright (c) 2011, RidgeRun
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the RidgeRun.
 * 4. Neither the name of the RidgeRun nor the
 *    names of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY RIDGERUN ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL RIDGERUN BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef JETSONGPIO_H_
#define JETSONGPIO_H_

 /****************************************************************
 * Constants
 ****************************************************************/
 
#define SYSFS_GPIO_DIR "/sys/class/gpio"
#define POLL_TIMEOUT (3 * 1000) /* 3 seconds */
#define MAX_BUF 64

typedef unsigned int jetsonGPIO ;
typedef unsigned int pinDirection ;
typedef unsigned int pinValue ;

enum pinDirections {
	inputPin  = 0,
	outputPin = 1
} ;

enum pinValues {
    low       = 0,
    high      = 1,
    off       = 0,  // synonym for things like lights
    on        = 1
}  ;


enum jetsonGPIONumber {
       jetson_pin50 = 57,   // J3A1 - Pin 50
	jetson_pin40 = 160,  // J3A2 - Pin 40	
	jetson_pin43 = 161,  // J3A2 - Pin 43
	jetson_pin46 = 162,  // J3A2 - Pin 46
	jetson_pin49 = 163,  // J3A2 - Pin 49
	jetson_pin52 = 164,  // J3A2 - Pin 52
	jetson_pin55 = 165,  // J3A2 - Pin 55
	jetson_pin58 = 166   // J3A2 - Pin 58
}  ;

enum jetsonTX1GPIONumber {
       jetsontx1_pin32 = 36,     // J21 - Pin 32 - Unused - AO_DMIC_IN_CLK
       jetsontx1_pin16 = 37,     // J21 - Pin 16 - Unused - AO_DMIC_IN_DAT
       jetsontx1_pin13 = 38,     // J21 - Pin 13 - Bidir  - GPIO20/AUD_INT
       jetsontx1_pin33 = 63,     // J21 - Pin 33 - Bidir  - GPIO11_AP_WAKE_BT
       jetsontx1_pin18 = 184,    // J21 - Pin 18 - Input  - GPIO16_MDM_WAKE_AP
       jetsontx1_pin31 = 186,    // J21 - Pin 31 - Input  - GPIO9_MOTION_INT
       jetsontx1_pin37 = 187,    // J21 - Pin 37 - Output - GPIO8_ALS_PROX_INT
       jetsontx1_pin29 = 219,    // J21 - Pin 29 - Output - GPIO19_AUD_RST
} ;


enum jetsonTX2GPIONumber {
       jetsontx2_pin32 = 297,    // J21 - Pin 32 - ??? - AO_DMIC_IN_CLK
       jetsontx2_pin16 = 296,    // J21 - Pin 16 - ??? - AO_DMIC_IN_DAT
       jetsontx2_pin13 = 397,    // J21 - Pin 13 - ??? - GPIO20/AUD_INT
       jetsontx2_pin33 = 389,    // J21 - Pin 33 - ??? - GPIO11_AP_WAKE_BT
       jetsontx2_pin18 = 481,    // J21 - Pin 18 - ??? - GPIO16_MDM_WAKE_AP
       jetsontx2_pin31 = 298,    // J21 - Pin 31 - ??? - GPIO9_MOTION_INT
       jetsontx2_pin37 = 388,    // J21 - Pin 37 - ??? - GPIO8_ALS_PROX_INT
       jetsontx2_pin29 = 398,    // J21 - Pin 29 - ??? - GPIO19_AUD_RST
} ;

enum jetsonXavierGPIONumber {
       jetsonxavier_pin32 = 257,    // J21 - Pin 32 - ??? - GPIO9_CAN1_GPIO0_DMIC_CLK
       jetsonxavier_pin16 = 256,    // J21 - Pin 16 - ??? - GPIO8_AO_DMIC_IN_DAT
       jetsonxavier_pin13 = 424,    // J21 - Pin 13 - ??? -     PWM01
       jetsonxavier_pin33 = 248,    // J21 - Pin 33 - ??? -     CAN1_DOUT
       jetsonxavier_pin18 = 344,    // J21 - Pin 18 - ??? -     GPIO35_PWM3
       jetsonxavier_pin31 = 250,    // J21 - Pin 31 - ??? -     CAN0_DOUT
       jetsonxavier_pin37 = 249,    // J21 - Pin 37 - ??? -     CAN1_DIN
       jetsonxavier_pin29 = 251,    // J21 - Pin 29 - ??? -     CAN0_DIN
} ;

enum jetsonNanoGPIONumber {
       jetsonnano_pin32 = 168,    // J21 - Pin 32 - ??? -     LCD_BL_PWM
       jetsonnano_pin16 = 232,    // J21 - Pin 16 - ??? -     SPI_2_CS1
       jetsonnano_pin13 = 14,     // J21 - Pin 13 - ??? -     SPI_2_SCK
       jetsonnano_pin33 = 38,     // J21 - Pin 33 - ??? -     GPIO_PE6
       jetsonnano_pin18 = 15,     // J21 - Pin 18 - ??? -     SPI_2_CS0
       jetsonnano_pin31 = 200,    // J21 - Pin 31 - ??? -     GPIO_PZ0
       jetsonnano_pin37 = 12,     // J21 - Pin 37 - ??? -     SPI_2_MOSI
       jetsonnano_pin29 = 149,    // J21 - Pin 29 - ??? -     CAM_AF_EN
} ;

int gpioExport ( jetsonGPIO gpio ) ;
int gpioUnexport ( jetsonGPIO gpio ) ;
int gpioSetDirection ( jetsonGPIO, pinDirection out_flag ) ;
int gpioSetValue ( jetsonGPIO gpio, pinValue value ) ;
int gpioGetValue ( jetsonGPIO gpio, unsigned int *value ) ;
int gpioSetEdge ( jetsonGPIO gpio, char *edge ) ;
int gpioOpen ( jetsonGPIO gpio ) ;
int gpioClose ( int fileDescriptor ) ;
int gpioActiveLow ( jetsonGPIO gpio, unsigned int value ) ;



#endif // JETSONGPIO_H_
