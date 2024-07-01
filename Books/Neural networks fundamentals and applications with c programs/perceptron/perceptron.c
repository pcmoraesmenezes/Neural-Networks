#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ENTRY 3
#define OUTPUT 2
#define IN 4

void clrscr() {
    system("clear");
}

void initialize_weights(float w[ENTRY][OUTPUT]) {
    for (int x = 0; x < ENTRY; x++) {
        for (int cont = 0; cont < OUTPUT; cont++) {
            w[x][cont] = 0.0;
        }
    }
}

float compute_net_input(float w[ENTRY][OUTPUT], float bias, float entrys[IN][ENTRY-1], int contin, int output) {
    float ni = w[0][output] * bias;
    for (int contt = 0; contt < ENTRY - 1; contt++) {
        ni += w[contt + 1][output] * entrys[contin][contt];
    }
    return ni;
}

float activation_function(float ni, int function) {
    switch (function) {
        case 1: 
            return (ni > 0) ? 1.0 : 0.0;
        case 2: 
            return 1.0 / (1.0 + exp(-ni));
        default:
            return 0.0;
    }
}

int main() {
    float w[ENTRY][OUTPUT], error[OUTPUT], ni[OUTPUT], bias, eta, entrys[IN][ENTRY-1], outputs[IN][OUTPUT], phi[OUTPUT];
    float err, errorm;
    int epochs, function;

    clrscr();

    initialize_weights(w);

    printf("Enter the value of bias: ");
    scanf("%f", &bias);

    printf("Enter the learning ratio (eta): ");
    scanf("%f", &eta);

    printf("Enter the number of epochs: ");
    scanf("%d", &epochs);

    printf("Enter the desired error: ");
    scanf("%f", &err);

    printf("Enter the desired function [(1) step, (2) sigmoid]: ");
    scanf("%d", &function);

    for (int x = 0; x < IN; x++) {
        for (int cont = 0; cont < ENTRY - 1; cont++) {
            printf("Entry %d, Neuron %d: ", x + 1, cont + 1);
            scanf("%f", &entrys[x][cont]);
        }
        for (int cont = 0; cont < OUTPUT; cont++) {
            printf("Output %d, Neuron %d: ", x + 1, cont + 1);
            scanf("%f", &outputs[x][cont]);
        }
    }

    printf("All initial weights are set to 0.\n");
    printf("Starting iterations...\n");

    int cont = 0;
    int testerror = 0;
    while (cont < epochs && !testerror) {
        clrscr();
        cont++;

        printf("Iteration: %d\n", cont);

        for (int x = 0; x < IN; x++) {
            printf("Entry: %d\n", x + 1);

            for (int output = 0; output < OUTPUT; output++) {
                ni[output] = compute_net_input(w, bias, entrys, x, output);
                phi[output] = activation_function(ni[output], function);

                error[output] = outputs[x][output] - phi[output];
                printf("Desired output: %f\n", outputs[x][output]);
                printf("Net output: %f\n", phi[output]);
            }

            errorm = 0.0;
            for (int output = 0; output < OUTPUT; output++) {
                errorm += fabs(error[output]); 
            }
            errorm /= OUTPUT; 
            printf("Mean general error: %f\n", errorm);

            testerror = (errorm < err) ? 1 : 0;

            printf("Adjusting weights...\n");
            for (int output = 0; output < OUTPUT; output++) {
                w[0][output] += eta * error[output] * bias;
                for (int contt = 0; contt < ENTRY - 1; contt++) {
                    w[contt + 1][output] += eta * error[output] * entrys[x][contt];
                }
            }

            printf("Updated weights:\n");
            for (int i = 0; i < ENTRY; i++) {
                for (int j = 0; j < OUTPUT; j++) {
                    printf("w[%d][%d] = %f\n", i, j, w[i][j]);
                }
            }
        }


    }

    printf("Done!\n");
    return 0;
}
