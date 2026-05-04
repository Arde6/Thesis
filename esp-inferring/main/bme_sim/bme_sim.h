#ifndef BME_SIM_H
#define BME_SIM_H

typedef struct {
    bool full;
    float gas_data[10];
    int gas_index;
    int step_index;
    float temp_data;
    float pres_data;
    float humi_data;
} bme_sim_data_t;

extern bme_sim_data_t sim_data;

void add_bme_data(float gas, float temp, float pres, float humi, int step);

void empty_bme_data();

#endif