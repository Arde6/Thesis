/**
 * @file mutex.h
 * 
 * @brief Init mutexs and variables used for data with mutex. Also includes a mutex creation function for contexts
 */

#ifndef MUTEX_H
#define MUTEX_H

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

extern float temperature;
extern float humidity;
extern float pressure;
extern float flow;

extern SemaphoreHandle_t bmeMutex;
extern SemaphoreHandle_t sfmMutex;
extern SemaphoreHandle_t i2cMutex;
extern SemaphoreHandle_t mqttMutex;

/*
 * @brief Initialize mutexes for shared resources
 *
 * @warning Call this function before MQTT and I2C device initialization
 */
int mutex_init(void);

/**
 * @brief Function to create mutexes for contexts
 */
SemaphoreHandle_t create_mutex(void);

#endif // MUTEX_H
