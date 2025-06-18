#ifndef ESP32_CSI_SYNC_COMPONENT_H
#define ESP32_CSI_SYNC_COMPONENT_H

#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_log.h"
#include <arpa/inet.h>
#include "time_component.h"
#include <string.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

// Broadcast MAC address for ESP-NOW sync packets
static const uint8_t sync_peer_mac[ESP_NOW_ETH_ALEN] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

#ifdef CONFIG_WIFI_CHANNEL
#define SYNC_WIFI_CHANNEL CONFIG_WIFI_CHANNEL
#else
#define SYNC_WIFI_CHANNEL 0
#endif

static inline uint64_t htonll(uint64_t val) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((uint64_t)htonl((uint32_t)(val >> 32)) |
            ((uint64_t)htonl((uint32_t)val) << 32));
#else
    return val;
#endif
}

#define ntohll htonll

typedef enum {
    SYNC_MSG_OFFSET = 1,
    SYNC_MSG_EPOCH  = 2
} sync_msg_type_t;

typedef struct __attribute__((packed)) {
    uint8_t  type;
    uint32_t seq;
    int64_t  timestamp_us;
} sync_packet_t;

typedef struct __attribute__((packed)) {
    uint8_t  type;
    int64_t tv_sec;
    int32_t tv_usec;
} time_sync_msg_t;

// ----------------------- MASTER -----------------------

static uint32_t sync_seq = 0;
static esp_timer_handle_t sync_timer;

static void send_sync_packet(void* arg)
{
    sync_packet_t pkt;
    pkt.type = SYNC_MSG_OFFSET;
    pkt.seq = sync_seq++;
    pkt.timestamp_us = esp_timer_get_time();
    esp_now_send(sync_peer_mac, (const uint8_t*)&pkt, sizeof(pkt));
}

static inline void sync_broadcast_epoch(const struct timeval *tv)
{
    time_sync_msg_t msg;
    msg.type = SYNC_MSG_EPOCH;
    msg.tv_sec = htonll(tv->tv_sec);
    msg.tv_usec = htonl(tv->tv_usec);
    esp_now_send(sync_peer_mac, (const uint8_t*)&msg, sizeof(msg));
    ESP_LOGI("SYNC_MASTER", "Broadcast epoch %lld.%06ld",
             (long long)tv->tv_sec, (long)tv->tv_usec);
}

static inline void sync_master_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, sync_peer_mac, ESP_NOW_ETH_ALEN);
    peer.channel = SYNC_WIFI_CHANNEL;
    peer.ifidx = WIFI_IF_STA;
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

static void sync_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len)
{
    if (len <= 0) {
        return;
    }

    uint8_t type = data[0];

    if (type == SYNC_MSG_OFFSET && len >= sizeof(sync_packet_t)) {
        sync_packet_t pkt;
        memcpy(&pkt, data, sizeof(pkt));
        int64_t local_time = esp_timer_get_time();
        time_offset = pkt.timestamp_us - local_time;
        ESP_LOGI("SYNC_WORKER", "Received seq %" PRIu32 ", offset=%lld us", pkt.seq, (long long)time_offset);
    } else if (type == SYNC_MSG_EPOCH && len >= sizeof(time_sync_msg_t)) {
        time_sync_msg_t msg;
        memcpy(&msg, data, sizeof(msg));
        msg.tv_sec = ntohll(msg.tv_sec);
        msg.tv_usec = ntohl(msg.tv_usec);
        struct timeval tv = { .tv_sec = msg.tv_sec, .tv_usec = msg.tv_usec };
        settimeofday(&tv, NULL);
        real_time_set = true;
        ESP_LOGI("SYNC_WORKER", "Epoch set to %lld.%06ld", (long long)msg.tv_sec, (long)msg.tv_usec);
    }
}

static inline void sync_worker_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_now_init());
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
