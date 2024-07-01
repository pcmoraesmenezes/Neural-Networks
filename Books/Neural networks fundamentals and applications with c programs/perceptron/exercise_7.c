#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define ENTRY 3 // 2 inputs (genre, author) + 1 bias
#define OUTPUT 2 // 2 classes (Engineering, Literature)
#define IN 6 // Number of training samples

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

float compute_net_input(float w[ENTRY][OUTPUT], float bias, int genre, int author, int output) {
    float ni = w[0][output] * bias + w[1][output] * genre + w[2][output] * author;
    return ni;
}

float activation_function(float ni, int function) {
    switch (function) {
        case 1: 
            return (ni > 0) ? 1.0 : 0.0; // Step function
        case 2: 
            return 1.0 / (1.0 + exp(-ni)); // Sigmoid function
        default:
            return 0.0;
    }
}

int main() {
    float w[ENTRY][OUTPUT], error[OUTPUT], ni[OUTPUT], bias, eta, phi[OUTPUT];
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

    // Training data: genre (0 for Engineering, 1 for Literature), author (0 for Woman, 1 for Man)
    int books[IN][2] = {
        {0, 1}, // Engineering book by a man
        {0, 1}, // Engineering book by a man
        {0, 1}, // Engineering book by a man
        {0, 0}, // Engineering book by a woman
        {1, 1}, // Literature book by a man
        {1, 0}  // Literature book by a woman
    };

    // Expected outputs (Engineering: Class 0, Literature: Class 1)
    int outputs[IN] = {0, 0, 0, 0, 1, 1};

    printf("All initial weights are set to 0.\n");
    printf("Starting iterations...\n");

    int cont = 0;
    int testerror = 0;
    while (cont < epochs && !testerror) {
        clrscr();
        cont++;

        printf("Iteration: %d\n", cont);

        for (int x = 0; x < IN; x++) {
            int genre = books[x][0];
            int author = books[x][1];

            printf("Book: %d\n", x + 1);
            printf("Genre: %s, Author: %s\n", (genre == 0) ? "Engineering" : "Literature", (author == 1) ? "Man" : "Woman");

            for (int output = 0; output < OUTPUT; output++) {
                ni[output] = compute_net_input(w, bias, genre, author, output);
                phi[output] = activation_function(ni[output], function);

                error[output] = (output == outputs[x]) ? 1 - phi[output] : -phi[output];
                printf("Desired output: %d\n", outputs[x]);
                printf("Net output: %f\n", phi[output]);
            }

            errorm = fabs(error[outputs[x]]);
            printf("Mean general error: %f\n", errorm);

            testerror = (errorm < err) ? 1 : 0;

            printf("Adjusting weights...\n");
            for (int output = 0; output < OUTPUT; output++) {
                w[0][output] += eta * error[output] * bias;
                w[1][output] += eta * error[output] * genre;
                w[2][output] += eta * error[output] * author;
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
