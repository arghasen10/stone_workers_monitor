#include <ESPmDNS.h>
#include "library/Wifi.h"
#include"library/pol.h"
#include <HTTPClient.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h> 
int nodeID = 7; 
char syncedTime[32] = "01-01-1970 00:00:00";

char tempStr[64];
WIFI wls = WIFI();
bool enabled = false;
AsyncWebServer server(80);
PollutantSensors pol=PollutantSensors(0x08,nodeID);
unsigned long lastSyncTime = 0;
const unsigned long syncInterval = 5000;
const char* serverName = "http://10.5.20.240:6001/sensor"; 

unsigned long lastClockUpdate = 0;
unsigned long clockInterval = 1000;
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

void request_sync_time() {
    WiFiClient client;
    HTTPClient http;
    String timeServer = "http://10.5.20.240:6001/time"; 

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
            pol.setSyncedTime(syncedTime);
            sscanf(syncedTime, "%d-%d-%d %d:%d:%d", &day, &month, &year, &hours, &minutes, &seconds);
        } else {
            Serial.print("JSON parse error: ");
            Serial.println(error.c_str());
        }
        
    } else {
        Serial.print("Failed to get time. HTTP code: ");
        Serial.println(httpResponseCode);
    }

    http.end();
}

void handle_sync(AsyncWebServerRequest *request) {
    if (request->hasParam("time", true)) {
        String timeParam = request->getParam("time", true)->value();
        timeParam.toCharArray(syncedTime, 32);  
        Serial.print("Time synchronized: ");
        Serial.println(syncedTime);
        // sscanf(syncedTime, "%d-%d-%d %d:%d:%d", &day, &month, &year, &hours, &minutes, &seconds);
        request->send_P(200, "text/html", "Time Set");
    } else {
        request->send_P(400, "text/html", "Missing time parameter");
    }
}

void updateClock() {
    if (millis() - lastClockUpdate >= clockInterval) {
        lastClockUpdate = millis();
        seconds++;
        if (seconds >= 60) {
            seconds = 0;
            minutes++;
        }
        if (minutes >= 60) {
            minutes = 0;
            hours++;
        }
        if (hours >= 24) {
            hours = 0;
            day++;  
        }

        Serial.printf("Current Time: %02d-%02d-%d %02d:%02d:%02d\n", day, month, year, hours, minutes, seconds);
        snprintf(syncedTime, sizeof(syncedTime), "%02d-%02d-%d %02d:%02d:%02d", 
                 day, month, year, hours, minutes, seconds);

        pol.setSyncedTime(syncedTime);
    }
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
    server.on("/sync", HTTP_POST, [&](AsyncWebServerRequest * request) {handle_sync(request);});
    pol.setSyncedTime(syncedTime);
    Serial.print("Time synchronized: ");
    
    Serial.println(syncedTime);
    server.begin();
    
}

void loop() {
    wls.Maintain();
    pol.Sample(Serial2,Serial);
    updateClock();
    if (millis() - lastSyncTime > syncInterval) {
        lastSyncTime = millis();
        request_sync_time();
    }
    if (enabled) {
        // sprintf(tempStr, "Id=%d&Temp=%d%d.%d&Hum=%d%d.%d&PM2_5=%d%d%d&Time=%s", 
        //         nodeID, 10, 10, 10, 10, 10, 10, 10, 10, 10, syncedTime);
        Serial.println(pol.reading);
        post_request(pol.reading);
    }
    delay(1000);
}

