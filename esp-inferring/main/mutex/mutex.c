/**
 * @file mutex.c
 * 
 * @brief Init mutexs and variables used for data with mutex. Also includes a mutex creation function for contexts
 */

#include "mutex.h"
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

float temperature = 0.0;
float humidity = 0.0;
float pressure = 0.0;
float flow = 0.0;

SemaphoreHandle_t bmeMutex;
SemaphoreHandle_t sfmMutex;
SemaphoreHandle_t i2cMutex;
SemaphoreHandle_t mqttMutex;

int mutex_init(void) {
    int rslt = 0;
    i2cMutex = xSemaphoreCreateMutex();
    if (i2cMutex == NULL) {
        printf("I2C Mutex creation failed!\n");
        rslt = -1;
    }
    bmeMutex = xSemaphoreCreateMutex();
    if (bmeMutex == NULL) {
        printf("BME Mutex creation failed!\n");
        rslt = -1;
    }
    sfmMutex = xSemaphoreCreateMutex();
    if (sfmMutex == NULL) {
        printf("SFM Mutex creation failed!\n");
        rslt = -1;
    }
    mqttMutex = xSemaphoreCreateMutex();
    if (mqttMutex == NULL) {
        printf("MQTT Mutex creation failed!\n");
        rslt = -1;
    }

    return rslt;
}

//Redundant, but for consitencys sake here
SemaphoreHandle_t create_mutex(void) {
    SemaphoreHandle_t mutex = xSemaphoreCreateMutex();
    return mutex;
}
