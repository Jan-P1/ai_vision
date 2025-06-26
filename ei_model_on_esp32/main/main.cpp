#include <stdio.h>
#include "esp_camera.h"

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

#include "driver/gpio.h"
#include "sdkconfig.h"
#include "esp_idf_version.h"

#define LED_PIN GPIO_NUM_4

//Camera config für ESP32. values genommen aus https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/
camera_config_t camera_config = {
    .pin_pwdn       = 32,
    .pin_reset      = -1,  // Set to -1 if not used
    .pin_xclk       = 0,
    .pin_sscb_sda   = 26,
    .pin_sscb_scl   = 27,
    
    .pin_d7         = 35,
    .pin_d6         = 34,
    .pin_d5         = 39,
    .pin_d4         = 36,
    .pin_d3         = 21,
    .pin_d2         = 19,
    .pin_d1         = 18,
    .pin_d0         = 5,
    .pin_vsync      = 25,
    .pin_href       = 23,
    .pin_pclk       = 22,

    .xclk_freq_hz   = 20000000,
    .ledc_timer     = LEDC_TIMER_0,
    .ledc_channel   = LEDC_CHANNEL_0,

    .pixel_format   = PIXFORMAT_RGB565,
    .frame_size     = FRAMESIZE_96X96,     
    .jpeg_quality   = 12,
    .fb_count       = 1
};

//features, bestehend aus raw data aus dem camera input
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {};

void setup_led() {
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
    esp_rom_gpio_pad_select_gpio(LED_PIN);
#elif ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(4, 0, 0)
    gpio_pad_select_gpio(LED_PIN);
#endif
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
}

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

void print_inference_result(ei_impulse_result_t result) {
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    //print befehle sind hartes bottleneck und sollten, falls kein serial monitor benötigt wird rausgenommen werden
    ei_printf("PREDICTIONS_START\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("%s:%.3f\n", ei_classifier_inferencing_categories[i], result.classification[i].value);
    }
    //led ansteuern wenn dimitrij erkannt wird
    if(result.classification[0].value > 0.6f) {
        gpio_set_level(LED_PIN, 1);
    }else{
        gpio_set_level(LED_PIN, 0);
    }
    ei_printf("PREDICTIONS_END\n");
}

extern "C" int app_main()
{
    setup_led();
    ei_sleep(100);
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ei_printf("Camera init failed with error 0x%x\n", err);
        return 1;
    }

    ei_impulse_result_t result = { nullptr };

    ei_printf("Edge Impulse standalone inferencing (Espressif ESP32)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
    {
        ei_printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        return 1;
    }

    ei_printf("Running inference on %d features...\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    while (true)
    {
        camera_fb_t * fb = esp_camera_fb_get();
        if (!fb) {
            ei_printf("Camera capture failed\n");
            continue;
        }

        // Make sure there's enough data
        int copy_len = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
        if (fb->len < copy_len) {
            ei_printf("Frame too small: got %d bytes, expected %d\n", fb->len, copy_len);
            esp_camera_fb_return(fb);
            continue;
        }

        ei_printf("FRAME_START\n");
        int feature_ix = 0;
            for (int i = 0; i < fb->len; i += 2) {
                uint16_t pixel = (fb->buf[i] << 8) | fb->buf[i + 1];

                uint8_t r = ((pixel >> 11) & 0x1F) << 3;
                uint8_t g = ((pixel >> 5) & 0x3F) << 2;
                uint8_t b = (pixel & 0x1F) << 3;
                uint8_t rgb[3] = { r, g, b };
                fwrite(rgb, 1, 3, stdout);
                uint32_t rgb888 = (r << 16) | (g << 8) | b;
                features[feature_ix++] = rgb888;
                
                if (feature_ix >= 19200) break;
            }
        ei_printf("FRAME_END\n");

        esp_camera_fb_return(fb);

        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false); //letzter parameter ist debug parameter, auf true setzen um feature inhalt den der classifier bekommt zu sehen
        if (res != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", res);
            return res;
        }

        print_inference_result(result);
        ei_sleep(100);
    }
}

