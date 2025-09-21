int main() {
  volatile unsigned char * data=(unsigned char*) 0x512;
  *data=10;
   
  return 0;
}
