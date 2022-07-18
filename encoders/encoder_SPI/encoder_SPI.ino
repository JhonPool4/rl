#include <SPI.h>
#define SENSORS_NUMBER 1
#define MAX_COUNT 65536.0

int32_t result[SENSORS_NUMBER] = {0};          // {0, 0, 0, 0, 0, 0};
const int chipSelectors[SENSORS_NUMBER] = {9}; // {5,6,7,8,9,10};
int cycles = 0;

String data_enc1;

int32_t v_angle[2] = {0};

long val_enc1 = 0;
float enc1 = 0.0;

int first_loop = 0;

void setup() {

  // initialize serial communication
  Serial.begin(115200);
  SPI.begin();


  for ( int aux = 0; aux < SENSORS_NUMBER; aux++ ) {
    pinMode(chipSelectors[aux], OUTPUT);
    delay(10);
    digitalWrite (chipSelectors[aux], HIGH);
  }


}

void loop() {

  // LEITURA ENCODERS //
  for ( int aux = 0; aux < SENSORS_NUMBER; aux++ )
  {
    // Initializes the SPI bus using the defined SPISettings.
    // SPISettings mySetting(speedMaximum, dataOrder, dataMode)
    // speedMaximum: maximum speed of communication
    // dataOrder: MSBFIRST or LSBFIRST
    // dataMode: SPI_MODE0, SPI_MODE1, SPI_MODE2, or SPI_MODE3
    SPI.beginTransaction(SPISettings(1000, MSBFIRST, SPI_MODE1) );
    digitalWrite (chipSelectors[aux], LOW);
    
    v_angle[1] = v_angle[0];
    v_angle[0] = SPI.transfer16(0x00); // address | value;

    
    
    if((-v_angle[0] + v_angle[1]) >= (MAX_COUNT*3.0/4.0)) cycles+=1;
    else if((-v_angle[0] + v_angle[1]) <= (-MAX_COUNT*3.0/4.0)) cycles-=1;
        
    
    digitalWrite (chipSelectors[aux], HIGH);
    SPI.endTransaction();

    if(first_loop<=10) first_loop++;

    if(first_loop > 10){
      Serial.print("angle:\t");
      Serial.println(enc1);
    }
    enc1 = (v_angle[0] + cycles*MAX_COUNT) * 360.0 / 65536.0 ;
    //Serial.println(atan2(sin(a),cos(a)));
    //Serial.println(enc1);
    
    
   
  }
  delay(50);

  /*data_enc1 = String(result[0], BIN);
  char copy_enc1[17];
  data_enc1.toCharArray(copy_enc1, 17);
  val_enc1 = strtol(copy_enc1, NULL, 2);
  
  //enc1 = val_enc1 ;*/

}
