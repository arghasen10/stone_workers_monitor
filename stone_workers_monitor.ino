#include <ESPmDNS.h>
#include "library/Wifi.h"
#include <HTTPClient.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

char tempStr[64];
WIFI wls = WIFI();
bool enabled = false;
AsyncWebServer server(80);

const char* serverName = "http://10.5.20.190:6001/sensor";

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

void handle_sync(AsyncWebServerRequest *request) {
    request->send_P(200, "text/html", "Time Set");
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
    wls.Connect();

    // Start mDNS with a unique hostname
    if (!MDNS.begin("nodemcu-sensor")) {
        Serial.println("Error setting up MDNS responder!");
        return;
    }
    Serial.println("mDNS responder started");

    Serial.println(WiFi.localIP());
    server.on("/con", HTTP_GET, [](AsyncWebServerRequest * request) {handle_con(request);});
    server.on("/coff", HTTP_GET, [](AsyncWebServerRequest * request) {handle_coff(request);});
    server.on("/sync", HTTP_POST, [&](AsyncWebServerRequest * request) {handle_sync(request);});
    server.begin();
}

void loop() {
    wls.Maintain();
    if (enabled) {
        sprintf(tempStr, "Id=%d&Temp=%d%d.%d&Hum=%d%d.%d&PM2_5=%d%d%d", 1, 10, 10, 10, 10, 10, 10, 10, 10, 10);
        Serial.println(tempStr);
        post_request(tempStr);
    }
    delay(1000);
}