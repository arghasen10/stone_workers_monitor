#ifndef DUST_H
#define DUST_H

class DUST{
    private:
        unsigned char buffer_RTT[40];
    
    public:
        unsigned int PMS1,PMS2_5,PMS10,FMHDS,TPS,HDS;
        DUST(void);
        void ReadData(Stream &port);
        void Sample(Stream &port);
        unsigned int deCodeBuffAt(int idx1,int idx2);
        void process32(Stream &port);
        void process40(Stream &port);
};

DUST::DUST(){
    PMS1 = 0;PMS2_5 =0;PMS10 = 0;FMHDS = 0;TPS = 0;HDS = 0;
}

unsigned int DUST::deCodeBuffAt(int idx1,int idx2){
    unsigned int val = 0,bytea = 0,byteb = 0;
    bytea=buffer_RTT[idx1];
    byteb=buffer_RTT[idx2];
    val=(bytea<<8)+byteb;
    return val;
}

void DUST::ReadData(Stream &port){
    unsigned int MODE = 0;

    while(!port.available());
    while(port.available()>0){
        buffer_RTT[0]=(char)port.read();delay(2);
        buffer_RTT[1]=(char)port.read();delay(2);
        buffer_RTT[2]=(char)port.read();delay(2);
        buffer_RTT[3]=(char)port.read();delay(2);

        MODE=deCodeBuffAt(2,3);

        if(MODE==28){
            process32(port);
        }else if(MODE==36){
            process40(port);
        }
    }
}

void DUST::Sample(Stream &port){
    ReadData(port);
}

void DUST::process32(Stream &port){
    unsigned int CR1 = 0,CR2 = 0;

    for(int i=4;i<32;i++){
        buffer_RTT[i]=(char)port.read();
        delay(2);
    }
    port.flush();

    //----------Checksum
    CR1 =deCodeBuffAt(30,31);
    for(int i=0;i<30;i++){
        CR2 += buffer_RTT[i];
    }
    //---------------//
    
    if(CR1 == CR2){
        PMS1=deCodeBuffAt(10,11);
        PMS2_5=deCodeBuffAt(12,13);
        PMS10=deCodeBuffAt(14,15);
        FMHDS=-1; //HCHO is not present in new sensors
        TPS=deCodeBuffAt(24,25);
        HDS=deCodeBuffAt(26,27);
    }
    else{
        PMS1 = 0;PMS2_5 =0;PMS10 = 0;FMHDS = 0;TPS = 0;HDS = 0;
    }
}

void DUST::process40(Stream &port){
    unsigned int CR1 = 0,CR2 = 0;

    for(int i=4;i<40;i++){
        buffer_RTT[i]=(char)port.read();
        delay(2);
    }
    port.flush();

    //----------Checksum
    CR1 =deCodeBuffAt(38,39);
    for(int i=0;i<38;i++){
        CR2 += buffer_RTT[i];
    }
    //---------------//
    
    if(CR1 == CR2){
        PMS1=deCodeBuffAt(10,11);
        PMS2_5=deCodeBuffAt(12,13);
        PMS10=deCodeBuffAt(14,15);
        FMHDS=deCodeBuffAt(28,29);
        TPS=deCodeBuffAt(30,31);
        HDS=deCodeBuffAt(32,33);
    }
    else{
        PMS1 = 0;PMS2_5 =0;PMS10 = 0;FMHDS = 0;TPS = 0;HDS = 0;
    }
}

#endif