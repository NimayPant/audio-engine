#ifndef BENCHMARK_H
#define BENCHMARK_H

namespace benchmark {

void runAllBenchmarks(int bufferSize = 1048576, int iterations = 100);
void printCpuFeatures();

}

#endif
