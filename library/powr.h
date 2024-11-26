#ifndef POWR_H
#define POWR_H

void softResetAtPowerUP(){
    esp_reset_reason_t reason = esp_reset_reason();
    if(reason==ESP_RST_POWERON){esp_restart();}
}

#endif