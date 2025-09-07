int main() {
    volatile int * data=(int*) 0x80000512;
    for (int i=0;i<10;i++) {
        data[i]=(i*100)/2;
    }

    return 0;
}
