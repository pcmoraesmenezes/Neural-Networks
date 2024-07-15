#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ncesc 8
#define input 2
#define output 1
#define example 4

void clrscr() {
    system("clear");
}

float target_function(int x1, int x2) {
    return x1 ^ x2;
}

int main() {
    float w[ncesc][output], W[input][ncesc], errodes, Erroinst, Erromg = 0, erro[output],
    niesc[ncesc], ni[output], biasesc[ncesc], biass[output], eta, phiesc[ncesc], phi[output], philesc[ncesc], phil[output], delta[output], deltaesc[ncesc];

    int x, y, cont2, contt, epochs, func;
    float entrys[input][example], outputs[output][example];

    clrscr();

    printf("Generating training data...\n");
    int xor_inputs[example][input] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for(y = 0; y < example; y++) {
        entrys[0][y] = xor_inputs[y][0];
        entrys[1][y] = xor_inputs[y][1];
        outputs[0][y] = target_function(xor_inputs[y][0], xor_inputs[y][1]);
    }

    printf("Initial weights and biases...\n");

    // Initialize weights and biases
    for(y = 0; y < ncesc; y++){
        for(x = 0; x < input; x++){
            W[x][y] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
        }
        for(x = 0; x < output; x++){
            w[y][x] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
        }
        biasesc[y] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
    }

    for(x = 0; x < output; x++){
        biass[x] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
    }

    printf("Enter the number of epochs:\n");
    scanf("%d", &epochs);

    printf("Enter the learning rate (eta):\n");
    scanf("%f", &eta);

    printf("Enter the desired error:\n");
    scanf("%f", &errodes);

    printf("Enter the desired function [(1) step, (2) sigmoid, (3) tanh]:\n");
    scanf("%d", &func);

    clrscr();

    printf("Starting iterations...\n");
    for(x = 0; x < epochs; x++){
        for (y = 0; y < example; y++){
            for(contt = 0; contt < ncesc; contt++){
                niesc[contt] = 0;
                for(cont2 = 0; cont2 < input; cont2++){
                    niesc[contt] += W[cont2][contt] * entrys[cont2][y];
                }
                niesc[contt] += biasesc[contt];

                switch(func){
                    case 1:
                        phiesc[contt] = (niesc[contt] > 0) ? 1.0 : 0.0;
                        break;
                    case 2:
                        phiesc[contt] = 1.0 / (1.0 + exp(-niesc[contt]));
                        break;
                    case 3:
                        phiesc[contt] = tanh(niesc[contt]);
                        break;
                    default:
                        phiesc[contt] = 0.0;
                }
            }

            for(contt = 0; contt < output; contt++){
                ni[contt] = 0;
                for (cont2 = 0; cont2 < ncesc; cont2++){
                    ni[contt] += w[cont2][contt] * phiesc[cont2];
                }
                ni[contt] += biass[contt];

                switch(func){
                    case 1:
                        phi[contt] = (ni[contt] > 0) ? 1.0 : 0.0;
                        break;
                    case 2:
                        phi[contt] = 1.0 / (1.0 + exp(-ni[contt]));
                        break;
                    case 3:
                        phi[contt] = tanh(ni[contt]);
                        break;
                    default:
                        phi[contt] = 0.0;
                }
            }

            for(contt = 0; contt < output; contt++){
                erro[contt] = outputs[contt][y] - phi[contt];
            }

            Erroinst = 0;
            for(contt = 0; contt < output; contt++){
                Erroinst += erro[contt] * erro[contt] / 2;
            }

            Erromg = (Erromg * (x * example + y) + Erroinst) / (x * example + y + 1);
            if (Erromg < errodes){
                printf("Desired error achieved. Exiting training.\n");
                x = epochs; // Force exit from outer loop
                break;
            }

            // Log the real output vs network output
            printf("Iteration %d, Real output: %f, Network output: %f, Error: %f\n", x * example + y, outputs[0][y], phi[0], erro[0]);

            for(cont2 = 0; cont2 < output; cont2++){
                if (func == 3) {
                    phil[cont2] = 1 - phi[cont2] * phi[cont2];
                } else {
                    phil[cont2] = exp(-ni[cont2]) / pow(1 + exp(-ni[cont2]), 2);
                }
                delta[cont2] = -erro[cont2] * phil[cont2];
            }

            for(cont2 = 0; cont2 < ncesc; cont2++){
                if (func == 3) {
                    philesc[cont2] = 1 - phiesc[cont2] * phiesc[cont2];
                } else {
                    philesc[cont2] = exp(-niesc[cont2]) / ((1 + exp(-niesc[cont2])) * (1 + exp(-niesc[cont2])));
                }
                deltaesc[cont2] = 0;

                for(contt = 0; contt < output; contt++){
                    deltaesc[cont2] += delta[contt] * w[cont2][contt];
                }
                deltaesc[cont2] *= philesc[cont2];
            }

            for(cont2 = 0; cont2 < output; cont2++){
                for(contt = 0; contt < ncesc; contt++){
                    w[cont2][contt] -= eta * delta[cont2] * phiesc[contt];
                }
                biass[cont2] -= eta * delta[cont2];
            }

            for(cont2 = 0; cont2 < ncesc; cont2++){
                for(contt = 0; contt < input; contt++){
                    W[cont2][contt] -= eta * deltaesc[cont2] * entrys[contt][y];
                }
                biasesc[cont2] -= eta * deltaesc[cont2];
            }

            if(Erromg < errodes){
                break;
            }
        }
        printf("Epoch %d, Mean squared error: %f\n", x, Erromg);

        if (Erromg < errodes) {
            printf("Desired error achieved after epoch %d. Exiting training.\n", x);
            break;
        }
        printf("End of epoch %d\n", x);
    }

    printf("Final biases:\n");
    for(y = 0; y < output; y++){
        printf("Output neuron %d: %f\n", y, biass[y]);
    }

    printf("Final weights:\n");
    for(y = 0; y < ncesc; y++){
        for(x = 0; x < input; x++){
            printf("Hidden neuron %d, input neuron %d: %f\n", y, x, W[x][y]);
        }
        for(x = 0; x < output; x++){
            printf("Output neuron %d, hidden neuron %d: %f\n", x, y, w[y][x]);
        }
    }

    printf("Training finished. Final mean squared error: %f\n", Erromg);

    return 0;
}