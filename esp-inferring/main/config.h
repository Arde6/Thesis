#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>
#include <stdbool.h>

// MQTT Topics
#define INFERRING_TOPIC    "inferring/test"    // Inferring

// MQTT config in config.c
extern const char *mqtt_broker;         // Broker ip
extern const char *mqtt_username;       // Username if any, NOT IN USE
extern const char *mqtt_password;       // Password if any, NOT IN USE
extern const int mqtt_port;             // Broker port
extern bool mqtt_connected;             // MQTT connection status

// IP address
extern char esp_ip[16];

// Mac address
extern uint8_t mac_address[6];
extern char mac_str[18];

// Timings
#define BME_DELAY pdMS_TO_TICKS(140)                        // Delay in ms between measurements
#define SFM_DELAY pdMS_TO_TICKS(500)                        // Delay in ms between measurements
#define DISPLAY_RATE pdMS_TO_TICKS(1000)                    // Delay in ms between display updates
#define MUTEX_TIMEOUT pdMS_TO_TICKS(200)                    // Timeout for Mutex
#define MQTT_TIMEOUT pdMS_TO_TICKS(1000)                    // Timeout for MQTT semaphore
#define MQTT_PUBLISH_INTERVAL_MS pdMS_TO_TICKS(500)         // Interval in ms between MQTT publishes
#define STRETCH_TIMEOUT pdMS_TO_TICKS(5000)                 // Timeout for bitbang do while loops
#define DEVICE_INFO_PUBLISH_INTERVAL pdMS_TO_TICKS(1000)    // Interval in ms between device info publishes
#define DEVICE_SCAN_INTERVAL pdMS_TO_TICKS(1000)            // Delay in ms between device scanning

#endif // CONFIG_H