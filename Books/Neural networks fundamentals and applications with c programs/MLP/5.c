#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ncesc 8
#define input 2
#define output 2
#define example 10 // Adjust this as needed for the number of training examples
#define M_PI 3.14159265358979323846

void clrscr() {
    system("clear");
}

float function1(float x, float z) {
    return 3 * x + z;
}

float function2(float x, float z) {
    return 5 * x * x - z;
}

int main() {
    float w[ncesc][output], W[input][ncesc], prev_w_update[ncesc][output], prev_W_update[input][ncesc];
    float errodes, Erroinst, Erromg = 0, erro[output],
    niesc[ncesc], ni[output], biasesc[ncesc], biass[output], prev_biasesc_update[ncesc], prev_biass_update[output];
    float eta, momentum, phiesc[ncesc], phi[output], philesc[ncesc], phil[output], delta[output], deltaesc[ncesc];

    int x, y, cont2, contt, epochs, func;
    float entrys[input][example], outputs[output][example];

    clrscr();

    printf("Generating training data...\n");
    for(y = 0; y < example; y++) {
        entrys[0][y] = ((float)rand() / RAND_MAX) * 10 - 5; // Random x in range [-5, 5]
        entrys[1][y] = ((float)rand() / RAND_MAX) * 10 - 5; // Random z in range [-5, 5]
        outputs[0][y] = function1(entrys[0][y], entrys[1][y]);
        outputs[1][y] = function2(entrys[0][y], entrys[1][y]);
    }

    printf("Initial weights and biases...\n");

    // Initialize weights and biases
    for(y = 0; y < ncesc; y++){
        for(x = 0; x < input; x++){
            W[x][y] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
            prev_W_update[x][y] = 0.0;
        }
        for(x = 0; x < output; x++){
            w[y][x] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
            prev_w_update[y][x] = 0.0;
        }
        biasesc[y] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
        prev_biasesc_update[y] = 0.0;
    }

    for(x = 0; x < output; x++){
        biass[x] = ((float)rand() / (float)(RAND_MAX)) * 2 - 1;
        prev_biass_update[x] = 0.0;
    }

    printf("Enter the number of epochs:\n");
    scanf("%d", &epochs);

    printf("Enter the learning rate (eta):\n");
    scanf("%f", &eta);

    printf("Enter the momentum factor:\n");
    scanf("%f", &momentum);

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
                        phiesc[contt] = (float)(1.0 / (1.0 + exp(-niesc[contt])));
                        break;
                    case 3:
                        phiesc[contt] = (float)tanh(niesc[contt]);
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
                        phi[contt] = (float)(1.0 / (1.0 + exp(-ni[contt])));
                        break;
                    case 3:
                        phi[contt] = (float)tanh(ni[contt]);
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

            Erromg = (Erromg * (x * example + y) + Erroinst) / (float)(x * example + y + 1);
            if (Erromg < errodes){
                printf("Desired error achieved. Exiting training.\n");
                x = epochs; // Force exit from outer loop
                break;
            }

            // Log the real output vs network output
            printf("Iteration %d, Real outputs: [%f, %f], Network outputs: [%f, %f], Errors: [%f, %f]\n", x * example + y, outputs[0][y], outputs[1][y], phi[0], phi[1], erro[0], erro[1]);

            for(cont2 = 0; cont2 < output; cont2++){
                if (func == 3) {
                    phil[cont2] = 1 - phi[cont2] * phi[cont2];
                } else {
                    phil[cont2] = (float)(exp(-ni[cont2]) / pow(1 + exp(-ni[cont2]), 2));
                }
                delta[cont2] = -erro[cont2] * phil[cont2];
            }

            for(cont2 = 0; cont2 < ncesc; cont2++){
                if (func == 3) {
                    philesc[cont2] = 1 - phiesc[cont2] * phiesc[cont2];
                } else {
                    philesc[cont2] = (float)(exp(-niesc[cont2]) / ((1 + exp(-niesc[cont2])) * (1 + exp(-niesc[cont2]))));
                }
                deltaesc[cont2] = 0;

                for(contt = 0; contt < output; contt++){
                    deltaesc[cont2] += delta[contt] * w[cont2][contt];
                }
                deltaesc[cont2] *= philesc[cont2];
            }

            for(cont2 = 0; cont2 < output; cont2++){
                for(contt = 0; contt < ncesc; contt++){
                    float w_update = eta * delta[cont2] * phiesc[contt] + momentum * prev_w_update[contt][cont2];
                    w[contt][cont2] -= w_update;
                    prev_w_update[contt][cont2] = w_update;
                }
                float bias_update = eta * delta[cont2] + momentum * prev_biass_update[cont2];
                biass[cont2] -= bias_update;
                prev_biass_update[cont2] = bias_update;
            }

            for(cont2 = 0; cont2 < ncesc; cont2++){
                for(contt = 0; contt < input; contt++){
                    float W_update = eta * deltaesc[cont2] * entrys[contt][y] + momentum * prev_W_update[contt][cont2];
                    W[contt][cont2] -= W_update;
                    prev_W_update[contt][cont2] = W_update;
                }
                float biasesc_update = eta * deltaesc[cont2] + momentum * prev_biasesc_update[cont2];
                biasesc[cont2] -= biasesc_update;
                prev_biasesc_update[cont2] = biasesc_update;
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
