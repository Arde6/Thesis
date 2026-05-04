#include "bme_sim.h"
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "config.h"
#include "ai.h"

#define TAG "BME"

bme_sim_data_t sim_data = {
    .full = false,
    .gas_index = 0,
};

void empty_bme_data() {
    sim_data.full = false;
    sim_data.gas_index = 0;
}

void add_bme_data(float gas, float temp, float pres, float humi, int step) {
    if (step != sim_data.gas_index) {
        ESP_LOGI(TAG, "Step not synced correctly!");
        sim_data.gas_index = 0;
        return;
    }
    if (sim_data.gas_index < 10) {
        sim_data.gas_data[sim_data.gas_index++] = gas;
    }
    
    if (sim_data.gas_index == 10) {
        sim_data.full = true;
        sim_data.temp_data = temp;
        sim_data.pres_data = pres;
        sim_data.humi_data = humi;

        xQueueSend(ai_queue, &sim_data, portMAX_DELAY);
        sim_data.gas_index = 0;
        sim_data.full = false;
    }
}

