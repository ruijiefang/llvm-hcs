#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{

  int b = atoi(argv[1]);
  
  long long j = 0;

  for(int i = 0; i < 1000000; ++i) {
    if (!b) {
      #include "chunk.txt"
      exit(1);
    } else {
      if (i % 2 == 0)
        j = j + 2;
      else if (i % 3 == 0)
        j = j + 1;
      else
        j = j - 2;
    }  
  }
  printf("%lld\n", j);
  return 0;
}

