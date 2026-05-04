#include "config.h"

#include "secrets.h"
#include "ai.h"
#include "mqtt.h"
#include "wifi.h"

#include <stdint.h>

// MQTT config
const char *mqtt_broker = MQTT_ADDRESS;                     // Broker ip, from secrets
const char *mqtt_username = NULL;                           // Username if any, NOT IN USE
const char *mqtt_password = NULL;                           // Password if any, NOT IN USE
const int mqtt_port = MQTT_PORT;                            // MQTT port, from secrets
bool mqtt_connected = false;      

// IP address variables
char esp_ip[16] = "0.0.0.0"; // Updated at wifi init
char esp_id[4] = "0";

// Mac address
uint8_t mac_address[6];
char mac_str[18];
