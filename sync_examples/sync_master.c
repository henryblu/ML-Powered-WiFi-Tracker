#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>

static const char *TAG = "SYNC_MASTER";
static uint8_t peer_mac[ESP_NOW_ETH_ALEN] = {0xff,0xff,0xff,0xff,0xff,0xff};

typedef struct {
    uint32_t seq;
    int64_t  timestamp_us;
} sync_packet_t;

static uint32_t seq = 0;

static void send_sync_packet(void* arg){
    sync_packet_t pkt;
    pkt.seq = seq++;
    pkt.timestamp_us = esp_timer_get_time();
    esp_now_send(peer_mac, (uint8_t*)&pkt, sizeof(pkt));
}

void app_main(void){
    esp_netif_init();
    esp_event_loop_create_default();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_start();

    esp_now_init();
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, peer_mac, ESP_NOW_ETH_ALEN);
    peer.channel = 0;
    peer.ifidx = ESP_IF_WIFI_STA;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    const esp_timer_create_args_t periodic_timer_args = {
            .callback = &send_sync_packet,
            .name = "sync_timer"};
    esp_timer_handle_t periodic_timer;
    esp_timer_create(&periodic_timer_args, &periodic_timer);
    esp_timer_start_periodic(periodic_timer, 1000000); // send every second

    ESP_LOGI(TAG, "Sync master started");
}
