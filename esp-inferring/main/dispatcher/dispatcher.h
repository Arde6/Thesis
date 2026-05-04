/**
 * @file dispatcher.c
 * 
 * @brief Dispatches data over MQTT and handles incoming messages
 */

#ifndef DISPATCHER_H
#define DISPATCHER_H

#include "mqtt_client.h"
#define DEVICE_DATA_DISPATCH_INTERVAL   1000
#define EXPECTED_PROFILE_SIZE           10
#define MAXIMUM_BME688_TEMPERATURE      500
#define MAXIMUM_BME688_MUL              500
#define BME688_BASE_DURATION            140

/**
 * @brief Dispatch device data over MQTT (device_data topic)
 */
void dispatchDeviceData(void *params);

/**
 * @brief Dispatch sensor data over MQTT (data topic)
 */
void dispatchSensorData(uint8_t i2c_addr, void *params);

/**
 * @brief Dispatch pump data over MQTT (fan_speed topic)
 */
void dispatchPumpData();

/**
 * @brief Handle incoming MQTT messages
 */
void dispatcherHandleMessage(esp_mqtt_event_handle_t event);

#endif
