#ifndef CO2_H
#define CO2_H

class CO2{
    private:
        int high,low,ch;
        unsigned char read_command[9];

    public:
        int co2;
        CO2(void);
        void NotifyNextRead(Stream &port);
        bool WaitforData(Stream &port);
        void ReadData(Stream &port);
        void Sample(Stream &port);
};

CO2::CO2(){
    high=0;low=0;co2=0;ch=0;
    unsigned char temp[]={0xFF,0x01,0x86,0x00,0x00,0x00,0x00,0x00,0x79};
    for(int i=0;i<9;i++){
      read_command[i]=temp[i];
    }
}

void CO2::NotifyNextRead(Stream &port){
    port.write(read_command,9);
}

bool CO2::WaitforData(Stream &port){
    while(true){
        if (port.available()>=9){return true;}
    }
}

void CO2::ReadData(Stream &port){
    for(int i=0;i<9;i++){
        ch=port.read();
        if(i==2){high=ch;}
        if(i==3){low=ch;}
    }
    co2=high*256+low;
}

void CO2::Sample(Stream &port){
    NotifyNextRead(port);
    WaitforData(port);
    ReadData(port);
}

#endif