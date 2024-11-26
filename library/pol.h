#ifndef HELPER_H
#define HELPER_H

#include"co2.h"
#include"dust.h"
#include"gas.h"
#include <cstring>


class PollutantSensors{
    private:
        CO2 co2_sensor;
        DUST dust_sensor;
        GAS gas_sensor;
        int nodeID;
        char syncedTimeval[64];
    public:
        char reading[512];

        PollutantSensors(uint8_t gas_addr, int nodeID);
        void Sample(Stream &dustport, Stream &co2port);
        void WriteToBuff();
        void setSyncedTime(const char *syncedTime);
};

PollutantSensors::PollutantSensors(uint8_t gas_addr, int nodeID)
    : co2_sensor(), dust_sensor(), gas_sensor(gas_addr), nodeID(nodeID) {
    // Use helper function to initialize syncedTimeval
}

void PollutantSensors::Sample(Stream &dustport, Stream &co2port){
    co2_sensor.Sample(co2port);
    dust_sensor.Sample(dustport);
    gas_sensor.Sample();
    WriteToBuff();
}

void PollutantSensors::setSyncedTime(const char *syncedTime) {
    strncpy(syncedTimeval, syncedTime, sizeof(syncedTimeval));
    syncedTimeval[sizeof(syncedTimeval) - 1] = '\0'; // Ensure null-termination
}

void PollutantSensors::WriteToBuff(){
  sprintf(reading,"{\"Id\":%d,\"Time\":\"%s\",\"T\":%d%d.%d,\"H\":%d%d.%d,\"FMHDS\":%d,\"PMS1\":%d,\"PMS2_5\":%d,\"PMS10\":%d,\"NO2\":%d,\"C2H5OH\":%d,\"VoC\":%d,\"CO\":%d,\"CO2\":%d}",
  nodeID, syncedTimeval,
  dust_sensor.TPS/100, (dust_sensor.TPS/10)%10, dust_sensor.TPS%10, 
  dust_sensor.HDS/100, (dust_sensor.HDS/10)%10, dust_sensor.HDS%10, 
  dust_sensor.FMHDS, 
  dust_sensor.PMS1,
  dust_sensor.PMS2_5,
  dust_sensor.PMS10,

  gas_sensor.NO2, 
  gas_sensor.C2H5CH, 
  gas_sensor.VoC, 
  gas_sensor.CO, 
  
  co2_sensor.co2);
}


#endif