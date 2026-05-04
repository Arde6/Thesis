/**
 * @file mqtt.c
 * 
 * @brief MQTT control for sending and recieving data to and from server/Dashboard
 * 
 * @see secrets.h and config files for client configurations.
 */ 

#include "mqtt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/FreeRTOSConfig_arch.h"
#include "freertos/idf_additions.h"
#include "freertos/task.h"
#include "mqtt_client.h"
#include "config.h"
#include "esp_log.h"
#include <cJSON.h>
#include "mutex.h"
#include "dispatcher.h"

static const char *TAG = "MQTT";
esp_mqtt_client_handle_t client = NULL;
static QueueHandle_t mqttQueue;
static TaskHandle_t mqttHandle;
//Subscribe to what we want. Number after topic = QoS
static esp_mqtt_topic_t topic_subscription_list[] = {
    {INFERRING_TOPIC,          0},        
};


static esp_err_t esp_mqtt_event_handler_cb(esp_mqtt_event_handle_t event) {
    esp_mqtt_client_handle_t client = event->client;
    
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected");
            mqtt_connected = true;

            // Subscribe to topic list, perhaps generate it based on connected i2c devices before calling this?
            esp_mqtt_client_subscribe_multiple(client, topic_subscription_list, (sizeof(topic_subscription_list) / sizeof(topic_subscription_list[0])));
            // ESP_LOGI(TAG, "Subscribed to pump topic, msg_id=%d", msg_id_1);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT disconnected");
            mqtt_connected = false;
            break;
        case MQTT_EVENT_PUBLISHED:
            // ESP_LOGI(TAG, "MQTT message published, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_SUBSCRIBED:
            // ESP_LOGI(TAG, "MQTT subscribed");
            break;
        case MQTT_EVENT_UNSUBSCRIBED:
            // ESP_LOGI(TAG, "MQTT unsubscribed");
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "Error occured");
            break;
        case MQTT_EVENT_DATA:
            // ESP_LOGI(TAG, "MQTT_EVENT_DATA");
            // printf("TOPIC=%.*s\r\n", event->topic_len, event->topic);
            // printf("DATA=%.*s\r\n", event->data_len, event->data);
            
            // Callback handler under dispatcher to avoid this ballooning in size here
            dispatcherHandleMessage(event);
            break;
        default:
            break;
    }   
    return ESP_OK;
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_mqtt_event_handler_cb((esp_mqtt_event_handle_t)event_data);
}

int mqtt_init(void) {

    int rslt;

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = mqtt_broker,
        .broker.address.port = mqtt_port,
        // .credentials.username = mqtt_username,
        // .credentials.password = mqtt_password,
        // .network.refresh_connection_after_ms = 60000,
    };

    client = esp_mqtt_client_init(&mqtt_cfg);

    if (client == NULL) {
        ESP_LOGW(TAG, "mqtt_init failed to create a client\n");
        return -1;
    }

    rslt = esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, client);
    if (rslt != ESP_OK) {
        ESP_LOGW(TAG, "esp_mqtt_client_register_event failed \n");
    }

    rslt = esp_mqtt_client_start(client);

    if (rslt != ESP_OK) {
        ESP_LOGW(TAG, "esp_mqtt_client_start failed \n");
    }

    mqttQueue = xQueueCreate(20, sizeof(MqttMessage));
    if (mqttQueue == NULL) {
        ESP_LOGE(TAG, "Failed to create mqtt message queue\n");
        rslt = 1;
        return rslt;
    }
    
    xTaskCreate(mqtt_publisher, "MQTT Publisher", configMINIMAL_STACK_SIZE + 4096, NULL, tskIDLE_PRIORITY + 1, &mqttHandle);

    return rslt;
}

void mqtt_publisher(void *params) {
    MqttMessage msg;
    while (1) {
        if (xQueueReceive(mqttQueue, &msg, portMAX_DELAY) == pdTRUE) {
            if (xSemaphoreTake(mqttMutex, portMAX_DELAY) == pdTRUE) {
                bool ok = esp_mqtt_client_publish(client, msg.topic, msg.msg, 0, 1, 0);
                if (!ok) {
                    ESP_LOGW(TAG, "MQTT publish failed: topic=%s\n", msg.topic);
                }
                xSemaphoreGive(mqttMutex);
            }
        }
    }
}

int send_message(const char* topic, const char* data) {
    if (mqttQueue == NULL) return -1;

    MqttMessage msg;
    if (strlen(topic) >= sizeof(msg.topic)) {
        ESP_LOGW(TAG, "Attempted topic field overflow \n");
        return -1;
    }


    if (strlen(data) >= sizeof(msg.msg)) {
        ESP_LOGW(TAG, "Attempted data field overflow \n");
        return -1;
    }

    strncpy(msg.topic, topic, sizeof(msg.topic) - 1);
    msg.topic[sizeof(msg.topic) - 1] = '\0';
    
    strncpy(msg.msg, data, sizeof(msg.msg) - 1);
    msg.msg[sizeof(msg.msg) - 1] = '\0';

    if (xQueueSend(mqttQueue, &msg, 0) != pdTRUE) {
        ESP_LOGW(TAG, "MQTT queue full, dropping message for %s\n", topic);
        return -1;
    }

    return 0;
}

int mqtt_set_broker(const char *new_uri, int new_port) {
    int rslt = 0;
    if (mqttMutex == NULL) {
        ESP_LOGW(TAG, "mqttMutex not initialized?\n");
        rslt = -1;
        return rslt;
    }

    if (xSemaphoreTake(mqttMutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        ESP_LOGW(TAG, "Failed to acquire mutex\n");
        rslt = -1;
        return rslt;
    }

    if (mqttHandle) vTaskSuspend(mqttHandle);
    
    if (client) {
        esp_mqtt_client_stop(client);
        esp_mqtt_client_destroy(client);
        client = NULL;
        if (mqttQueue) {
            xQueueReset(mqttQueue);
        }
    }

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = new_uri,
        .broker.address.port = new_port,
        // .credentials.username = mqtt_username,
        // .credentials.password = mqtt_password,
        .network.refresh_connection_after_ms = 60000,
    };

    client = esp_mqtt_client_init(&mqtt_cfg);

    if (!client) {
       ESP_LOGE(TAG, "Failed to create a new mqttClient\n");
        xSemaphoreGive(mqttMutex);
        if (mqttHandle) vTaskResume(mqttHandle);
        return -1;
    }

    rslt = esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, client);

    if (rslt != ESP_OK) {
        ESP_LOGW(TAG, "esp_mqtt_client_register_event failed \n");
    }

    if (rslt != ESP_OK) {
        ESP_LOGW(TAG, "esp_mqtt_client_start failed \n");
    }

    if (mqttHandle) vTaskResume(mqttHandle);
    xSemaphoreGive(mqttMutex);
    return rslt;
}
