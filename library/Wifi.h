#ifndef WIfi_H
#define WIfi_H

#include <WiFi.h>
#include <WiFiMulti.h>
#include "esp_wpa2.h"

#define EAP_ANONYMOUS_IDENTITY "" //anonymous identity
#define EAP_IDENTITY "22CS91R03"  //user identity
#define EAP_PASSWORD "bittu123"   //user password

class WIFI{
    private:
        WiFiMulti wifiMulti;
    public:
        WIFI(void);
        void AddSSID(const char *ssid, const char *password);
        void Connect(void);
        void Maintain(void);
};

WIFI::WIFI(){
    wifiMulti.addAP("SMR_LAB", "smrl1991");
    // wifiMulti.addAP("B2", "123#456#");
}

void WIFI::AddSSID(const char *ssid, const char *password){
    wifiMulti.addAP(ssid, password);
}

void WIFI::Connect(){
    int count=0,flag=0;
    while(wifiMulti.run() != WL_CONNECTED){
        if(count%10==0){
            WiFi.disconnect(true);
            WiFi.mode(WIFI_STA);
            if(flag==0){
                flag=1;
            }else{
                esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EAP_ANONYMOUS_IDENTITY, strlen(EAP_ANONYMOUS_IDENTITY));
                esp_wifi_sta_wpa2_ent_set_username((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
                esp_wifi_sta_wpa2_ent_set_password((uint8_t *)EAP_PASSWORD, strlen(EAP_PASSWORD));
                esp_wifi_sta_wpa2_ent_enable();
                flag=0;
            }
            
        }
        delay(1000);
        count++;
    }
}

void WIFI::Maintain(){
    Connect();
}

#endif