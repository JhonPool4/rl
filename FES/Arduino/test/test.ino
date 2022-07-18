float x=10;
void setup() {
  // put your setup code here, to run once:
Serial.begin(20000);

}

void loop() {
  // put your main code here, to run repeatedly:
  //float x=random(140)+10;
  x = x + .05;
  if (x>150){
    x = 0;
  }
  Serial.println(10);
}
