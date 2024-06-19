CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wcast-align \
         -Wconversion -Wsign-conversion -Wnull-dereference -g3 -O0
SRC = perceptron.c
TARGET = perceptron

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ -lm

	clean:
	rm -f $(TARGET)

.PHONY: all clean