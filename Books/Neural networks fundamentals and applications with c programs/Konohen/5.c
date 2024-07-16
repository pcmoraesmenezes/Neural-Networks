#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define nentry 2  // Two entries: one for y = x+1 and another for y = 2x
#define nneuron 3 // Number of neurons (arbitrarily chosen for simplicity)
#define nsample 10 // Number of samples for training

void clrscr() {
    system("@cls||clear");
}

int main() {
    float w[nentry][nneuron], errodes, tau, sigma0, sigma, eta, eta0;
    int x, y, cont2, contt, epochs;
    float entry[nentry][nsample], test[nentry][6], d[nneuron], l2, h, deucl;

    srand(time(NULL)); // Initialize the random number generator seed

    clrscr();

    // Initialize weights
    for (x = 0; x < nentry; x++)
        for (y = 0; y < nneuron; y++)
            w[x][y] = ((float)rand() / RAND_MAX) * 2 - 1; // Weights between -1 and 1

    printf("Number of epochs: \n");
    scanf("%d", &epochs);

    printf("Sample train vector (format: x for y=x+1 and y=2x): \n");
    for (x = 0; x < nentry; x++)
        for (y = 0; y < nsample; y++)
            scanf("%f", &entry[x][y]);

    printf("Initial learning rate: \n");
    scanf("%f", &eta0);

    printf("Desired error: \n");
    scanf("%f", &errodes);

    printf("Tau constant: \n");
    scanf("%f", &tau);

    clrscr();

    printf("Initial weights: \n");
    for (x = 0; x < nentry; x++) {
        for (y = 0; y < nneuron; y++)
            printf("w[%d][%d] = %f\n", x, y, w[x][y]);
    }

    printf("Iterative process: \n");

    for (x = 0; x < epochs; x++) {
        for (y = 0; y < nsample; y++) {
            for (contt = 0; contt < nneuron; contt++) {
                d[contt] = 0;
                for (cont2 = 0; cont2 < nentry; cont2++)
                    d[contt] += (entry[cont2][y] - w[cont2][contt]) * (entry[cont2][y] - w[cont2][contt]);
            }

            deucl = d[0];
            int winning_neuron = 0;
            for (contt = 0; contt < nneuron; contt++)
                if (deucl >= d[contt]) {
                    deucl = d[contt];
                    winning_neuron = contt;
                }

            if (deucl < errodes) break;

            for (contt = 0; contt < nneuron; contt++) {
                sigma = sigma0 * exp(-(x * nsample + y) / tau);
                l2 = pow((winning_neuron - contt), 2);
                h = exp(-l2 / (2 * sigma * sigma));

                eta = eta0 * exp(-(x * nsample + y) / tau);

                for (cont2 = 0; cont2 < nentry; cont2++)
                    w[cont2][contt] += eta * h * (entry[cont2][y] - w[cont2][contt]);
            }
        }

        if (deucl < errodes) break;
    }

    printf("Final weights: \n");
    for (contt = 0; contt < nentry; contt++)
        for (cont2 = 0; cont2 < nneuron; cont2++)
            printf("w[%d][%d] = %f\n", contt, cont2, w[contt][cont2]);

    printf("Test vector: \n");
    for (cont2 = 0; cont2 < 5; cont2++) {
        for (contt = 0; contt < nentry; contt++) {
            if (cont2 < 3) test[contt][cont2] = entry[contt][cont2];
            if (cont2 == 3) test[contt][cont2] = rand() % 20;
            if (cont2 == 4) {
                printf("Test vector: \n");
                scanf("%f", &test[contt][cont2]);
            }
        }
    }

    for (x = 0; x < 5; x++) {
        for (contt = 0; contt < nneuron; contt++) {
            d[contt] = 0;
            for (cont2 = 0; cont2 < nentry; cont2++)
                d[contt] += (test[cont2][x] - w[cont2][contt]) * (test[cont2][x] - w[cont2][contt]);
            printf("d = %f\n", d[contt]);
        }
        deucl = d[0];
        int winning_neuron = 0;
        for (contt = 0; contt < nneuron; contt++)
            if (deucl >= d[contt]) {
                deucl = d[contt];
                winning_neuron = contt;
            }

        printf("Entry:\n");
        for (y = 0; y < nentry; y++)
            printf("%f\n", test[y][x]);
        printf("Active Neuron: %d\n", winning_neuron + 1);
    }

    return 0;
}
