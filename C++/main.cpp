#include <bits/stdc++.h>
using namespace std;


class average {
public:
    double summer = 0;
    double count = 0;
    double avg = 0;
    void add(double other) {
        summer += other;
        count += 1;
        avg = summer / count;
    }
};


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_prime(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}


int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int n = 4;
    vector<int> lengths = {784, 16, 16, 10};
    int test_cases = 60000;
    int repeats = 10000;
    int batch_size = 1000;
    double learning_rate = 10;
    vector<double> C;

    double x;

    // testing data
    freopen("training_data", "r", stdin);
    vector<vector<vector<double>>> testing(test_cases, {vector<double>(lengths[0], 0), vector<double>(lengths[n - 1], 0)});
    for (int i = 0; i < test_cases; i++) {
        for (int j = 0; j < lengths[0]; j++) {
            cin >> testing[i][0][j];
        }
        for (int j = 0; j < lengths[n - 1]; j++) {
            cin >> testing[i][1][j];
        }
    }
    cout << "Testing data loaded!" << endl;

    // biases
    freopen("biases", "r", stdin);
    vector<vector<double>> biases(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < lengths[i]; j++) {
            cin >> x;
            biases[i].push_back(x);
        }
    }
    cout << "Biases loaded!" << endl;

    // weights
    freopen("weights", "r", stdin);
    vector<vector<vector<double>>> weights(n - 1, vector<vector<double>>());
    for (int i = 0; i < n - 1; i++) {
        weights[i].resize(lengths[i]);
        for (int k = 0; k < lengths[i]; k++) {
            weights[i][k].resize(lengths[i + 1]);
            for (int j = 0; j < lengths[i + 1]; j++) {
                cin >> weights[i][k][j];
            }
        }
    }
    cout << "Weights loaded!" << endl;

    // training
    freopen("log", "a", stdout);
    for (int repeat = 0; repeat < repeats; repeat++) {

        double batch_sum = 0.0;

        for (int batch = 0; batch < test_cases; batch += batch_size) {


            vector<vector<vector<average>>> gradient_weights(n - 1);
            for (int i = 0; i < n - 1; i++) {
                gradient_weights[i].resize(lengths[i]);
                for (int k = 0; k < lengths[i]; k++) {
                    gradient_weights[i][k].resize(lengths[i + 1]);
                }
            }
            vector<vector<average>> gradient_biases(n);
            for (int i = 0; i < n; i++) {
                gradient_biases[i].resize(lengths[i]);
            }
            average C_cur;

            for (int current_test = batch; current_test < batch + batch_size; current_test++) {
                vector<vector<double>> test = testing[current_test];
                vector<vector<double>> array(n);
                vector<vector<double>> z(n);
                for (int i = 0; i < n; i++) {
                    if (i == 0) {
                        array[i] = test[0];
                        z[i] = test[0];
                    } else {
                        array[i].resize(lengths[i]);
                        z[i].resize(lengths[i]);
                    }
                }
                vector<double> y = test[1];

                for (int i = 1; i < n; i++) {
                    for (int j = 0; j < lengths[i]; j++) {
                        for (int k = 0; k < lengths[i - 1]; k++) {
                            array[i][j] += array[i - 1][k] * weights[i - 1][k][j];
                        }
                        array[i][j] += biases[i][j];
                        z[i][j] = array[i][j];
                        array[i][j] = sigmoid(array[i][j]);
                    }
                }


                double C_cur_cur = 0;
                for (int i = 0; i < lengths[n - 1]; i++) {
                    C_cur_cur += pow(array[n - 1][i] - y[i], 2);
                }
                C_cur.add(C_cur_cur);


                vector<vector<double>> d_bias_z(n), d_array(n);
                for (int i = 0; i < n; i++) {
                    d_array[i].resize(lengths[i]);
                    d_bias_z[i].resize(lengths[i]);
                }
                vector<vector<vector<double>>> d_weights(n - 1);
                for (int i = 0; i < n - 1; i++) {
                    d_weights[i].resize(lengths[i]);
                    for (int k = 0; k < lengths[i]; k++) {
                        d_weights[i][k].resize(lengths[i + 1]);
                    }
                }

                for (int i = 0; i < lengths[n - 1]; i++) {
                    d_array[n - 1][i] = 2 * (array[n - 1][i] - y[i]);
                    d_bias_z[n - 1][i] = d_array[n - 1][i] * sigmoid_prime(z[n - 1][i]);
                }
                for (int i = n - 2; i >= 0; i--) {
                    for (int k = 0; k < lengths[i]; k++) {
                        for (int j = 0; j < lengths[i + 1]; j++) {
                            d_weights[i][k][j] = d_bias_z[i + 1][j] * array[i][k];
                            d_array[i][k] += d_bias_z[i + 1][j] * weights[i][k][j];
                        }
                        d_bias_z[i][k] = d_array[i][k] * sigmoid_prime(z[i][k]);
                    }
                }


                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < lengths[i]; j++) {
                        gradient_biases[i][j].add(d_bias_z[i][j]);
                    }
                }

                for (int i = 0; i < n - 1; i++) {
                    for (int k = 0; k < lengths[i]; k++) {
                        for (int j = 0; j < lengths[i + 1]; j++) {
                            gradient_weights[i][k][j].add(d_weights[i][k][j]);
                        }
                    }
                }
            }

            C.push_back(C_cur.avg);
            batch_sum += C_cur.avg;

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < lengths[i]; j++) {
                    biases[i][j] -= gradient_biases[i][j].avg * learning_rate;
                }
            }
            for (int i = 0; i < n - 1; i++) {
                for (int k = 0; k < lengths[i]; k++) {
                    for (int j = 0; j < lengths[i + 1]; j++) {
                        weights[i][k][j] -= gradient_weights[i][k][j].avg * learning_rate;
                    }
                }
            }


            freopen("log", "a", stdout);
            cout << batch << " " << repeat << " " << C_cur.avg << endl;

            if (batch == test_cases - batch_size) {
                cout << repeat << " " << batch_sum << " " << batch_sum / (test_cases / batch_size) << endl;
                batch_sum = 0.0;
            }

            if (batch == 0) {

                ofstream file;
                file.open("NN/trained" + to_string(repeat), ios::out);

                file << n << endl;
                for (int i: lengths) { file << i << " "; }
                file << endl;
                for (int i = 0; i < n - 1; i++) {
                    for (int j = 0; j < lengths[i]; j++) {
                        for (double k: weights[i][j]) { file << setprecision(20) << k << " "; }
                    }
                }
                file << endl;
                for (int i = 0; i < n; i++) {
                    for (double j: biases[i]) { file << setprecision(20) << j << " "; }
                }
                file << endl;

                file.close();
            }
        }
    }
    return 0;
}
