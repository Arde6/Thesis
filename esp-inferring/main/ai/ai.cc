#include "ai.h"
#include "bme_sim.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#define QUANT 0

#if QUANT == 1
#include "int8_stat_model.h"
#else
// #include "fp16_model.h"
#include "fp32_model.h"
#endif

static const char *TAG = "AI";

QueueHandle_t ai_queue;

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    constexpr int kTensorArenaSize = 512 * 1024;
    uint8_t* tensor_arena = nullptr;
}

void init_ai() {
    // Allocate Arena in PSRAM
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Could not allocate tensor arena in PSRAM!");
        return;
    }

    // Map the model
    #if QUANT == 1
    model = tflite::GetModel(int8_stat_model_tflite);
    #else
    // model = tflite::GetModel(models_fp16_model_tflite);
    model = tflite::GetModel(models_fp32_model_tflite);
    #endif

    // Check if model is compatible
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to suppoerted version %d.",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ESP_LOGI(TAG, "Model mapped");

    // Load Operations
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddSub();
    resolver.AddMul();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddAdd();

    // Initialize Interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate Tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed. Try increasing kTensorArenaSize");
        return;
    } 
    ESP_LOGI(TAG, "Tensors allocated");

    // Assign pointers to input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void ai_inference_task(void *pvParameters) {
    bme_sim_data_t received_data;
    
    while (1) {
        if (xQueueReceive(ai_queue, &received_data, portMAX_DELAY)) {
            // Take time to measure performance
            uint64_t start = esp_timer_get_time();

            // Transfer recieved data to ai input and scale if int8
            if (QUANT == 1) {
                float s = input->params.scale;
                int zp = input->params.zero_point;
                
                for (int i = 0; i < 10; i++) {
                    input->data.int8[i] = (int8_t)(received_data.gas_data[i] / s + zp);
                }
                
                input->data.int8[10] = (int8_t)(received_data.temp_data / s + zp);
                input->data.int8[11] = (int8_t)(received_data.pres_data / s + zp);
                input->data.int8[12] = (int8_t)(received_data.humi_data / s + zp);
            } else {
                for (int i = 0; i < 10; i++) {
                    input->data.f[i] = received_data.gas_data[i];
                }
                
                input->data.f[10] = received_data.temp_data;
                input->data.f[11] = received_data.pres_data;
                input->data.f[12] = received_data.humi_data;
            }
            
            // Check status
            TfLiteStatus invoke_status = interpreter->Invoke();

            // Predict
            if (invoke_status == kTfLiteOk) {
                int num_classes = output->dims->data[1]; 
                int predicted_class = 0;
                if (QUANT == 1) {
                    int8_t max_score = output->data.int8[0];
                    for (int i = 1; i < num_classes; i++) {
                        if (output->data.int8[i] > max_score) {
                            max_score = output->data.int8[i];
                            predicted_class = i;
                        }
                    }
                    ESP_LOGI(TAG, "Predicted Class: %d (Score: %d)", predicted_class, max_score);
                } else {
                    float max_score = output->data.f[0];
                    for (int i = 1; i < num_classes; i++) {
                        if (output->data.f[i] > max_score) {
                            max_score = output->data.f[i];
                            predicted_class = i;
                        }
                    }
                    ESP_LOGI(TAG, "Predicted Class: %d (Score: %.4f)", predicted_class, max_score);
                }
            } else {
                ESP_LOGE(TAG, "Inference failed!");
            }

            uint64_t end = esp_timer_get_time();
            uint64_t used_time = end - start;
            ESP_LOGI(TAG, "Used %lld microseconds", used_time);
        }
    }
}

void run_ai_logic() {
    // Create AI queue
    ai_queue = xQueueCreate(1, sizeof(bme_sim_data_t));

    // Init AI model
    init_ai();

    // Start inference task
    xTaskCreate(ai_inference_task, "AI Inference Task", 4096, NULL, 1, NULL);
}