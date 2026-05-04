#ifndef WIFI_H
#define WIFI_H

/*
 * Initialize WiFi in station mode and connect to the specified SSID.
 *
 * @warning Takes time so give it some delay after calling this function before starting any network operations.
 * 
 * @note Make sure to define WIFI_SSID and WIFI_PASS in secrets.h
 */
int wifi_init(void);

#endif // WIFI_H
