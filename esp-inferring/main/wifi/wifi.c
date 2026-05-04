/**
 * @file wifi.c
 * 
 * Wifi init, event handler and mdns
 * 
 */

#include "wifi.h"
#include "config.h"
#include <string.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "secrets.h"
#include <mdns.h>

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static const char *TAG = "wifi";

/**
 * @brief Start mDNS service for easier local network access
 * 
 * Hopefully this can just be called when it fails haha.
 * 
 * @warning If init fails it will just be logged and left at that
 * 
 * @return void
 */
void start_mdns_service(void) {
    esp_err_t err = mdns_init();
    if (err) {
        ESP_LOGE(TAG, "MDNS Init failed: %d", err);
        return;
    }

    char hostname[20];
    snprintf(hostname, sizeof(hostname), "esp-inferring");
    mdns_hostname_set(hostname);
    mdns_instance_name_set("Inferring");

    ESP_LOGI(TAG, "MDNS service started with hostname: %s.local", hostname);

    return;
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        ESP_LOGI(TAG, "Retrying connection to AP");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        esp_ip4addr_ntoa(&event->ip_info.ip, esp_ip, sizeof(esp_ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_CONNECTED) {
        start_mdns_service();
    }
}

int wifi_init(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "nvs_flash_init() failed\n");
        return -1;
    } 

    s_wifi_event_group = xEventGroupCreate();

    ret = esp_netif_init();

    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_netif_init() failed\n");
        return -1;
    } 
    ret = esp_event_loop_create_default();

    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_event_loop_create_default() failed\n");
        return -1;
    } 
    
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_init() failed\n");
        return -1;
    }

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ret = esp_event_handler_instance_register(WIFI_EVENT,
                                              ESP_EVENT_ANY_ID,
                                              &wifi_event_handler,
                                              NULL,
                                              &instance_any_id);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_event_handle_instance_register() WIFI_EVENT failed\n");
        return -1;
    }

    ret = esp_event_handler_instance_register(IP_EVENT,
                                              IP_EVENT_STA_GOT_IP,
                                              &wifi_event_handler,
                                              NULL,
                                              &instance_got_ip);

    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_event_handle_instance_register() IP_EVENT failed\n");
        return -1;
    }

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
            /* Setting a password implies WPA2. */
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ret = esp_wifi_set_mode(WIFI_MODE_STA);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_mode() failed\n");
        return -1;
    }
    ret = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_set_config() failed\n");
        return -1;
    }
    ret = esp_wifi_start();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "esp_wifi_start() failed\n");
        return -1;
    }

    ESP_LOGI(TAG, "wifi_init finished.");
    return 0;
}

