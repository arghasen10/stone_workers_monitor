#include <ESPmDNS.h>
#include "library/Wifi.h"
#include"library/pol.h"
#include <HTTPClient.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h> 
#include <ESP32Time.h>
#include "library/powr.h"


int nodeID = 3; 
char syncedTime[32] = "01-01-1970 00:00:00";
unsigned long lastSyncTime = 0;
unsigned long currTime=0;
const unsigned long syncInterval = 1000*60*20; //20 mins
ESP32Time rtc(0);
char timeString[64];
char tempStr[64];
WIFI wls = WIFI();
bool enabled = false;
AsyncWebServer server(80);
PollutantSensors pol=PollutantSensors(0x08,nodeID);
const char* serverName = "http://192.168.1.104:6001/sensor"; 

int seconds = 0, minutes = 0, hours = 0;
int day = 1, month = 1, year = 1970;

int post_request(String httpRequestData) {
    WiFiClient client;
    HTTPClient http;

    http.begin(client, serverName);

    http.addHeader("Content-Type", "application/x-www-form-urlencoded");
    int httpResponseCode = http.POST(httpRequestData);

    Serial.print("HTTP Response code: ");
    Serial.println(httpResponseCode);

    http.end();
    return httpResponseCode;
}

int request_sync_time() {
    WiFiClient client;
    HTTPClient http;
    String timeServer = "http://192.168.1.104:6001/time"; 

    http.begin(client, timeServer);
    int httpResponseCode = http.GET();

    if (httpResponseCode == 200) {
        String serverResponse = http.getString();
        Serial.print("Raw server response: ");
        Serial.println(serverResponse);
        StaticJsonDocument<128> jsonDoc; 
        DeserializationError error = deserializeJson(jsonDoc, serverResponse);
        if (!error) {
            const char* time = jsonDoc["time"]; 
            strncpy(syncedTime, time, sizeof(syncedTime)); 
            Serial.print("Time synchronized from server: ");
            Serial.println(syncedTime);
            sscanf(syncedTime, "%d-%d-%d %d:%d:%d", &day, &month, &year, &hours, &minutes, &seconds);
            rtc.setTime(seconds, minutes, hours, day, month, year);
            Serial.println("Time sync");
            return 1;
        } else {
            Serial.print("JSON parse error: ");
            Serial.println(error.c_str());
            return 0;
        }
        
    } else {
        Serial.print("Failed to get time. HTTP code: ");
        Serial.println(httpResponseCode);
        return 0;
    }

    http.end();
}


void handle_con(AsyncWebServerRequest *request) {
    enabled = true;
    request->send_P(200, "text/html", "Data Collection Started");
}

void handle_coff(AsyncWebServerRequest *request) {
    enabled = false;
    request->send_P(200, "text/html", "Data Collection Stopped");
}

void setup() {
    Serial.begin(9600);
    Serial2.begin(9600);
    softResetAtPowerUP();
    wls.Connect();
    char hostname[32];
    sprintf(hostname, "nodemcu-sensor%d", nodeID);
    Serial.print("HostName: ");Serial.println(hostname);
    // Start mDNS with a unique hostname
    if (!MDNS.begin(hostname)) {
        Serial.println("Error setting up MDNS responder!");
        return;
    }
    Serial.println("mDNS responder started");

    Serial.println(WiFi.localIP());
    server.on("/con", HTTP_GET, [](AsyncWebServerRequest * request) {handle_con(request);});
    server.on("/coff", HTTP_GET, [](AsyncWebServerRequest * request) {handle_coff(request);});
    server.begin();

    while (request_sync_time()==0){
      Serial.println("Retrying time sync");
    }
}

void loop() {
    wls.Maintain();
    currTime=millis();

    if (currTime- lastSyncTime > syncInterval) {
        lastSyncTime = millis();
        while (request_sync_time()==0){
          Serial.println("Retrying time sync");
        }
    }
    if(lastSyncTime> currTime){
      lastSyncTime=0;
    }
    if (enabled) {
        pol.Sample(Serial2,Serial);
        strncpy(timeString, rtc.getTime("%A, %B %d %Y %H:%M:%S").c_str(), sizeof(timeString));
        Serial.print("timevalue");Serial.println(timeString);
        pol.setSyncedTime(timeString);
        Serial.println(pol.reading);
        post_request(pol.reading);
    }
    delay(1000);
}

