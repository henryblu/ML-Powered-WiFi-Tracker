#ifndef ESP32_CSI_SYNC_COMPONENT_H
#define ESP32_CSI_SYNC_COMPONENT_H

#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_log.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Broadcast MAC address for ESP-NOW sync packets
static const uint8_t sync_peer_mac[ESP_NOW_ETH_ALEN] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

typedef struct {
    uint32_t seq;
    int64_t  timestamp_us;
} sync_packet_t;

// ----------------------- MASTER -----------------------

static uint32_t sync_seq = 0;
static esp_timer_handle_t sync_timer;

static void send_sync_packet(void* arg)
{
    sync_packet_t pkt;
    pkt.seq = sync_seq++;
    pkt.timestamp_us = esp_timer_get_time();
    esp_now_send(sync_peer_mac, (const uint8_t*)&pkt, sizeof(pkt));
}

static inline void sync_master_init(void)
{
    esp_now_init();

    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, sync_peer_mac, ESP_NOW_ETH_ALEN);
    peer.channel = 0;
    peer.ifidx = ESP_IF_WIFI_STA;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    const esp_timer_create_args_t periodic_timer_args = {
            .callback = &send_sync_packet,
            .name = "sync_timer"};
    esp_timer_create(&periodic_timer_args, &sync_timer);
    esp_timer_start_periodic(sync_timer, 1000000); // broadcast every second

    ESP_LOGI("SYNC_MASTER", "Sync master started");
}

// ----------------------- WORKER -----------------------

static int64_t time_offset = 0;

static void sync_recv_cb(const uint8_t *mac, const uint8_t *data, int len)
{
    if (len != sizeof(sync_packet_t)) {
        return;
    }
    sync_packet_t pkt;
    memcpy(&pkt, data, sizeof(pkt));
    int64_t local_time = esp_timer_get_time();
    time_offset = pkt.timestamp_us - local_time;
    ESP_LOGI("SYNC_WORKER", "Received seq %u, offset=%lld us", pkt.seq, (long long)time_offset);
}

static inline void sync_worker_init(void)
{
    esp_now_init();
    esp_now_register_recv_cb(sync_recv_cb);
    ESP_LOGI("SYNC_WORKER", "Sync worker started");
}

static inline int64_t get_synced_time(void)
{
    return esp_timer_get_time() + time_offset;
}

#ifdef __cplusplus
}
#endif

#endif // ESP32_CSI_SYNC_COMPONENT_H
