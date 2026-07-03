#include <bits/stdc++.h>
using namespace std;

#ifdef LOCAL
#include "debug.h"
#else
#define debug(...) 42
#endif

// Goal: teach a very simple neural network to double a floating point value in the range [0, 0.5]

// This neural network consists of one input, two hidden layers with one and two neurons, respectively, and one output. 


// Illustration below:
//                      --- O ---
//                   ----       ----
// O ----------- O ---             -------------- O
//                   ----       ----
//                      --- O ---

inline double sig(double n){ // sigmoid function keeps neuron activation between 0 and 1
    return 1 / (1 + exp(-n));
}

vector<pair<double,double>> trials = {
    {0.0, 0.0},
    {0.1, 0.2},
    {0.2, 0.4},
    {0.3, 0.6},
    {0.4, 0.8},
    {0.5, 1.0}
};

signed main() {
    double learning_rate = 1;
    double w1 = 0.5, w2 = -0.3, w3 = 0.7, w4 = 0.3, w5 = -0.2; // weights
    double b1 = 0.0, b2 = 0.0, b3 = 0.0, c = 0.0; // biases
    double firstCost, lastCost;
    int n = 1000000;
    for(int i = 0; i < n; i++){
        double totalCost = 0.0;
        for(auto [x, y] : trials){
            double h1 = sig(w1 * x + b1);
            double h2 = sig(w2 * h1 + b2);
            double h3 = sig(w3 * h1 + b3);
            double y1 = sig(w4 * h2 + w5 * h3 + c);
            double C = 0.5 * (y1 - y) * (y1 - y);
            totalCost += C;
            double d4 = (y1 - y) * y1 * (1 - y1); // manual backpropagation to make computation a bit easier (only saves a few operations because the network is so small)
            double pdc = d4, pdw5 = d4 * h3, pdw4 = d4 * h2; // partial derivatives with respect to c, w5, and w4
            double d3 = d4 * w5 * h3 * (1 - h3);
            double pdw3 = d3 * h1, pdb3 = d3; // partial derivatives with respect to w3 and b3
            double d2 = d4 * w4 * h2 * (1 - h2);
            double pdw2 = d2 * h1, pdb2 = d2; // partial derivatives with respect to w2 and b2
            double d1 = (d2 * w2 + d3 * w3) * h1 * (1 - h1);
            double pdw1 = d1 * x, pdb1 = d1; // partial derivatives with respect to w1 and b1
            w1 -= learning_rate * pdw1; 
            w2 -= learning_rate * pdw2; 
            w3 -= learning_rate * pdw3; 
            w4 -= learning_rate * pdw4; 
            w5 -= learning_rate * pdw5;
            b1 -= learning_rate * pdb1; 
            b2 -= learning_rate * pdb2; 
            b3 -= learning_rate * pdb3;
            c -= learning_rate * pdc;
        }
        if(i % 10 == 0) cout << "i: " << i << " --- Total Cost: " << totalCost << '\n'; // display the total cost across the data set every 10 trainings

        // To be noted here: 
        //
        // Change n to 1500 to see an interesting aspect of this training
        // In about the first 830 trainings, the total cost barely changes (starts at 0.375263, ends at 0.370363)
        // However, it changes drastically between the trainings 1000 and 1120 (starts at 0.330202, ends at 0.0218758)
        // This shows that sometimes training can be ineffective at first for many trials, but eventually it may become incredibly effective
        //
        // Also, at the start of the trials the cost actually goes up a bit, meaning that we may have jumped over a local minimum in the cost function because of a large enough learning rate
        // Sometimes a smaller learning rate is better because it can help avoid overshooting a local minimum
        // It could also mean that individual updates on individual examples from the dataset temporarily made the total dataset loss worse
        // In this case, it might actually be good that we used a greater learning rate as it might have got us out of a small dip in the function so that we could discover a large drop off
        // A larger drop off might have a local minimum with a much smaller value than the other local minimum we could have settled in
        //
        // Feel free to play around with the number of iterations and the learning rate to discover more interesting aspects of this cost function

        
    }
    // print final results after the last training set (should be close to doubling x each time)

    cout << "Final outputs:" << '\n';
    for (auto [x, y] : trials) {
        double h1 = sig(w1 * x + b1);
        double h2 = sig(w2 * h1 + b2);
        double h3 = sig(w3 * h1 + b3);
        double y1 = sig(w4 * h2 + w5 * h3 + c);

        cout << "x = " << x << "  target = " << y << "  output = " << y1 << '\n';
    }
    
}

