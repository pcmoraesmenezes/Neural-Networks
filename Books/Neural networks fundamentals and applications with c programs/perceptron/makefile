CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wcast-align \
         -Wconversion -Wsign-conversion -Wnull-dereference -g3 -O0
TARGETS = perceptron exercise_5 exercise_6 exercise_7

all: $(TARGETS)

perceptron: perceptron.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

exercise_5: exercise_5.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

exercise_6: exercise_6.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

exercise_7: exercise_7.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(TARGETS)

.PHONY: all clean
