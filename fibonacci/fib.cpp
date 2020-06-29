#include <iostream>


void fib(uint64_t n) {
    uint64_t a = 1;
    uint64_t b = 1;
    uint64_t tmp = 0;

    for (size_t i = 1; i <= n; i++) {
        std::cout << b << std::endl;
        tmp = a;
        a = a + b;
        b = tmp;
    }
}

int main(int argc, char** argv) {
   uint64_t n = std::stoi(argv[1]);

   fib(n);
   
   return 0;
}