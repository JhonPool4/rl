// documentation
// atmega3280: https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-7810-Automotive-Microcontrollers-ATmega328P_Datasheet.pdf
// THI40N: https://files.pepperl-fuchs.com/webcat/navi/productInfo/pds/t36797_eng.pdf?v=20220718111409

/************************
    enconder variables
 ************************/
uint8_t pin_m1a=7;    // interruption pin (phase A [Green])
uint8_t pin_m1b=6;    // interruption pin (phase B [Gray])

int32_t   global_counts_m1;
uint8_t   global_m1a_state;
uint8_t   global_m1b_state;
uint8_t   global_last_m1a_state;
uint8_t   global_last_m1b_state;



/************************
    headers
 ************************/
void enable_interrupts_pin(uint8_t pin);
void enconder_init(uint8_t pin_m1a, uint8_t pin_m1b);




void setup()
{
  enconder_init(pin_m1a, pin_m1b);
  Serial.begin(115200);
}

void loop() {

  Serial.println(global_counts_m1*360/2048);
  
}



/*******************************
    interrupt service routine
 *******************************/
ISR(PCINT2_vect){ /*handle pin change interrupt for PCINT[16-23] */

  // get state of each phase
  global_m1a_state = digitalRead(pin_m1a);
  global_m1b_state = digitalRead(pin_m1b);  

  bool forward_m1 = global_m1a_state ^ global_last_m1b_state; // clockwise direction
  bool backward_m1 = global_m1b_state ^ global_last_m1a_state; // counter clockwise direction
  
  
  if(forward_m1 == 1){      
    global_counts_m1 += 1;   
    //Serial.println("forward");       
  }
  
  if(backward_m1 == 1){
    global_counts_m1 -= 1;
    //Serial.println("backward");
  }

  // update last state of each phase
  global_last_m1a_state = global_m1a_state;
  global_last_m1b_state = global_m1b_state;
}


void enable_interrupts_pin(uint8_t pin){
  /* 
  Note: 
  ----
  - (Important!) This code just work for the arduino pins from 0 to 7.
  - PCIR: External Interrupt Control Register
  - DDRx: Destination Data Register x
  - PCMSK: Pin Change Mask Register
  */

  PCICR |= (1 << PCIE2);// enable pin change interrupt for the group PCIE2 (arduino: 0-7)
  DDRD &= ~(1 << pin);  // set the pin as input (0=input, 1=output)
  PCMSK2 |= (1 << pin); // select wheter pin is enable as interruption

}

void enconder_init(uint8_t pin_m1a, uint8_t pin_m1b){
  //disable interrupts  
  cli(); 

  // set the pins as interruptions
  enable_interrupts_pin(pin_m1a);
  enable_interrupts_pin(pin_m1b);
  
  // initialize the global state m1
  global_counts_m1 = 0;
  global_last_m1a_state = digitalRead(pin_m1a);
  global_last_m1b_state = digitalRead(pin_m1b);

  // Pin Change Interrupt Flag Register
  PCIFR |= (1 << PCIF0) | (1 << PCIF1) | (1 << PCIF2); // clear any outstanding interrupt

  //enable interrupts
  sei();
}
