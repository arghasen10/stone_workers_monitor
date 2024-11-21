#ifndef GAS_H
#define GAS_H

#include <Multichannel_Gas_GMXXX.h>
#include <Wire.h>

class GAS{
    private:
        bool status;
        uint8_t address;
        GAS_GMXXX<TwoWire> gas;
    public:
        int NO2,C2H5CH,VoC,CO;
        GAS(void);
        GAS(uint8_t addr);
        void begin(void);
        void Sample(void);
};

GAS::GAS(){}
GAS::GAS(uint8_t addr){
    status=false;
    address=addr;
    NO2=0;C2H5CH=0;VoC=0;CO=0;
}

void GAS::begin(){
    status=true;
    gas.begin(Wire,address);
}

void GAS::Sample(){
    if(status==false){begin();}

    NO2 = gas.getGM102B();
    if(NO2 > 999){NO2 = 999;}
    
    C2H5CH = gas.getGM302B();
    if(C2H5CH > 999){C2H5CH = 999;}
    
    VoC = gas.getGM502B();
    if(VoC > 999){VoC = 999;}
    
    CO = gas.getGM702B();
    if(CO > 999){CO = 999;}
}

#endif