#include "dispatcher.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/idf_additions.h"
#include "mqtt.h"
#include <cJSON.h>
#include <stddef.h>
#include <stdint.h>
#include "sys/param.h"
#include "config.h"
#include "bme_sim.h"

static const char *TAG = "DISPATCHER";

mqtt_topic_t identify_topic(const char* topic, size_t len) {
    // ESP_LOGI(TAG, "Identifying %.*s", (int)len, topic);
    const struct {
        const char* str;
        mqtt_topic_t type;
    } topic_map[] = {
        {INFERRING_TOPIC,          TOPIC_INFERRING    },
    };
    
    for (int i = 0; i < sizeof(topic_map)/sizeof(topic_map[0]); i++) {
        if (strlen(topic_map[i].str) == len && strncmp(topic, topic_map[i].str, len) == 0) {
            return topic_map[i].type;

        }
    }

    return TOPIC_UNKNOWN;
}

void dispatcherHandleMessage(esp_mqtt_event_handle_t event) {
    switch(identify_topic(event->topic, event->topic_len)) {
        case TOPIC_INFERRING:{
            char data_buf[2048];
            strncpy(data_buf, event->data, MIN(event->data_len, sizeof(data_buf) - 1));
            data_buf[MIN(event->data_len, sizeof(data_buf) - 1)] = '\0';
            // ESP_LOGI(TAG, "Inferring topic recieved (raw): %s", data_buf);

            cJSON *root = cJSON_Parse(data_buf);
            if (!root) {
                ESP_LOGE(TAG, "JSON parse error");
                return;
            }

            cJSON *gas = cJSON_GetObjectItem(root, "gas");
            cJSON *temp = cJSON_GetObjectItem(root, "temperature");
            cJSON *pres = cJSON_GetObjectItem(root, "pressure");
            cJSON *humi = cJSON_GetObjectItem(root, "humidity");
            cJSON *step = cJSON_GetObjectItem(root, "cycle_step_index");

            if (cJSON_IsNumber(gas) && cJSON_IsNumber(temp) && 
                cJSON_IsNumber(pres) && cJSON_IsNumber(humi)) {
                
                // Cast the double values to float
                add_bme_data((float)gas->valuedouble, 
                            (float)temp->valuedouble, 
                            (float)pres->valuedouble, 
                            (float)humi->valuedouble,
                            (int)step->valueint);
                            
            } else {
                ESP_LOGE(TAG, "Missing numeric fields in JSON");
            }

            cJSON_Delete(root);
            break;
        }
        default: {
            break;
        }
    }
};
