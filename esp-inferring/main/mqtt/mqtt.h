/**
 * @file mqtt.h
 * 
 * @brief MQTT control for sending and recieving data to and from server/Dashboard
 * 
 * @see secrets.h and config files for client configurations.
 */

#ifndef MQTT_H
#define MQTT_H 

/**
 * @brief MQTT Message struct
 * @note Increase stack allocation size per message if neccesary. Message queue is allocated dynamically based on MqttMessage size.
 * */
typedef struct {
    char topic[64];
    char msg[2048];
} MqttMessage;

/**
 * @brief Enumerator of topics for dispatcher switch case management
 */
typedef enum {
    TOPIC_INFERRING,
    TOPIC_UNKNOWN
} mqtt_topic_t;

/**
 * @brief Start MQTT client and publisher task
 * @note Call this function after WiFi is connected
 * @note This function creates a FreeRTOS task for publishing
 */

int mqtt_init (void); 


/*
 * @brief MQTT message publisher
 * */
void mqtt_publisher(void *params);


/*
 * @brief MQTT message queue for other components to write to
 * */
int send_message(const char* topic, const char* data);

/*
 * @brief Change broker on runtime
 * */
int mqtt_set_broker(const char *new_uri, int new_port);
#endif // MQTT_H


/**
 * @brief Publish BME data to MQTT broker
 * @note Call to publish pump speed to mqtt
 * @param speed Pump speed (0-100)
 * @todo Change pump speed to 0-255
 * @todo Add mac_address to topic {mac; speed}
 * @note Has internal mutex handling
 */
