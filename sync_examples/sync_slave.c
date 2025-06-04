#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>

static const char *TAG = "SYNC_SLAVE";
static int64_t time_offset = 0;

typedef struct {
    uint32_t seq;
    int64_t  timestamp_us;
} sync_packet_t;

static void recv_cb(const uint8_t *mac, const uint8_t *data, int len){
    if(len != sizeof(sync_packet_t)) return;
    sync_packet_t pkt;
    memcpy(&pkt, data, sizeof(pkt));
    int64_t local_time = esp_timer_get_time();
    time_offset = pkt.timestamp_us - local_time;
    ESP_LOGI(TAG, "Received seq %u, offset=%lld us", pkt.seq, (long long)time_offset);
}

int64_t get_synced_time(){
    return esp_timer_get_time() + time_offset;
}

void app_main(void){
    esp_netif_init();
    esp_event_loop_create_default();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_start();

    esp_now_init();
    esp_now_register_recv_cb(recv_cb);

    ESP_LOGI(TAG, "Sync slave started");
    while(true){
        ESP_LOGI(TAG, "Synced time %lld", (long long)get_synced_time());
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
