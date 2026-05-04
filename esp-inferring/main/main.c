#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "sdkconfig.h"

#include "ai.h"
#include "secrets.h"
#include "wifi.h"
#include "mutex.h"
#include "mqtt.h"
#include "config.h"

static const char *TAG = "main";

void app_main(void)
{
    ESP_LOGI(TAG, "Booted");
    
    mutex_init();
    wifi_init();

    while (mqtt_init() != 0) {
        vTaskDelay(2000 / portTICK_PERIOD_MS);
    }

    run_ai_logic();

    esp_read_mac(mac_address, ESP_MAC_EFUSE_FACTORY);
    snprintf(mac_str, sizeof(mac_str), "%02X:%02X:%02X:%02X:%02X:%02X", mac_address[0], mac_address[1],mac_address[2], mac_address[3],mac_address[4], mac_address[5]);

    printf("%s\n", mac_str);

    while (1) {
        vTaskDelay(5000 / portTICK_PERIOD_MS);
    }
}
