#include <iostream>

void foo(float* input, float* output, int size) {
    for (int index = 0; index < size; ++index) {
        output[index] = input[index] * 2.0f;  // Example computation
    }
}

int main() {
    int size = 1024;
    float* input = new float[size];
    float* output = new float[size];

    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(i);
    }

    foo(input, output, size);

    for (int i = 0; i < 10; ++i) {
        std::cout << output[i] << std::endl;
    }

    delete[] input;
    delete[] output;

    return 0;
}
