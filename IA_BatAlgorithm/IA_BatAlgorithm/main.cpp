#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using namespace std;

typedef int typeOptimization;
#define OPT_MAX 1
#define OPT_MIN 0
#define M_PI 3.1415926535897932384631

typedef int typeFunction;
#define FUNC_ACK 10
#define FUNC_SCH 11
#define FUNC_BE3 12


struct ConfigParams {
	int d,
		popSize,
		numIter;
	double r_ini,
		alpha, gamma, qMin, qMax, lower, upper;
	typeFunction funcType;
	typeOptimization optType;
};

double getRandValue() {
	return (rand() / (double)(RAND_MAX + 1));
}

void simpleBounds(vector<double> &vals, double lower, double upper) {
	int n = vals.size();
	for (int i = 0; i < n; i++) {
		if (vals[i] < lower)
			vals[i] = lower;
		else if (vals[i] > upper)
			vals[i] = upper;
	}
}

int bestBat(vector<double> fitness, typeOptimization type) {
	double currVal = fitness[0];
	int pos = 0;
	for (int i = 1; i < fitness.size(); i++) {
		if (currVal > fitness[i]) {
			currVal = fitness[i];
			pos = i;
		}
	}
	return pos;
}

double meanFromVector(vector<double> v) {
	double sum = 0;
	for (int i = 0; i < v.size(); i++) {
		sum += v[i];
	}
	return sum / v.size();
}

double functionAckley(vector<double> x) {
	double a = 20, b = 0.2;
	double c = 2 * M_PI;
	int d = x.size();
	double sum1 = 0, sum2 = 0;
	for (int i = 0; i < d; i++) {
		sum1 += x[i] * x[i];
		sum2 += cos(c * x[i]);
	}
	return -a * exp(-b * sqrt(sum1 / d)) - exp(sum2 / d) + a + exp(1);
}

double functionSchewel(vector<double> x) {
	int d = x.size();
	double sum1 = 0;
	for (int i = 0; i < d; i++) {
		sum1 += x[i] * sin(sqrt(abs(x[i])));
	}
	return 418.9829 * d - sum1;
}

double functionBenchmark3(vector<double> x) {
	double a = 0.5, b = 1, c = 0.001;
	int d = x.size();
	double sum1 = 0;
	for (int i = 0; i < d; i++) {
		sum1 += x[i] * x[i];
	}
	double nume = pow(sin(sqrt(sum1)), 2.0) - a;
	double deno = pow((b + c * sum1), 2.0);
	return a - nume / deno;
}

double funcObj(vector<double> x, typeFunction type, typeOptimization optType) {
	double val = 0;
	switch (type) {
	case FUNC_ACK:
		val = functionAckley(x);
		break;
	case FUNC_SCH:
		val = functionSchewel(x);
		break;
	case FUNC_BE3:
		val = functionBenchmark3(x);
		break;
	}
	if (optType == OPT_MAX)
		val *= -1;
	return val;
}

void BatFunction(ConfigParams params, vector<double> &bestSol, double &bestFit) {
	vector<double> A_loudness(params.popSize);
	vector<double> r_pulseRatio(params.popSize);
	vector<double> r_pulseRatioIni;
	vector<vector<double> > vel(params.popSize, vector<double>(params.d, 0));
	vector<vector<double> > velPos(params.popSize, vector<double>(params.d, 0));
	vector<vector<double> > x(params.popSize, vector<double>(params.d, 0));
	vector<vector<double> > xPos(params.popSize, vector<double>(params.d, 0));
	vector<double> fitnessByBat(params.popSize, 0);
	bestSol.clear();
	bestSol.resize(params.d, 0);
	bestFit = 0;

	srand(time(NULL));

	// Inicializando los valores
	for (int i = 0; i < params.popSize; i++) {
		A_loudness[i] = -1 + 2 * getRandValue();
		r_pulseRatio[i] = getRandValue();
		
		for (int j = 0; j < params.d; j++) {
			x[i][j] = params.lower + (params.upper - params.lower) * getRandValue();
		}
		fitnessByBat[i] = funcObj(x[i], params.funcType, params.optType);
	}
	r_pulseRatioIni = r_pulseRatio;

	int bestPos = bestBat(fitnessByBat, params.optType);
	bestSol = x[bestPos];
	bestFit = fitnessByBat[bestPos];

	double avgLoudness, randNu, freqI, randEpsilon, auxRand;

	// Empiezan las iteraciones
	for (int t = 0; t < params.numIter; t++) {
		for (int j = 0; j < params.popSize; j++) {
			randNu = getRandValue();
			freqI = params.qMin + (params.qMax - params.qMin) * randNu;

#pragma omp parallel for 
			for (int k = 0; k < params.d; k++) {
				velPos[j][k] = vel[j][k] + (x[j][k] - bestSol[k]) * freqI;
				xPos[j][k] = x[j][k] + velPos[j][k];
			}
			randNu = getRandValue();
			// Si se cumple la condicion, se mejora el X
			if (randNu > r_pulseRatio[j]) {
				avgLoudness = meanFromVector(A_loudness);
#pragma omp parallel for 
				for (int k = 0; k < params.d; k++) {
					randEpsilon = -1 + 2 * getRandValue();
					xPos[j][k] = bestSol[k] + randEpsilon * avgLoudness;
				}
			}
			// Se verifica que el valor este dentro del rango
			simpleBounds(xPos[j], params.lower, params.upper);
			// Se genera una nueva solucion
#pragma omp parallel for
			for (int k = 0; k < params.d; k++) {
				x[j][k] = params.lower + (params.upper - params.lower) * getRandValue();
			}

			// Evaluacion de convergencia
			if (randNu < A_loudness[j] && funcObj(xPos[j], params.funcType, params.optType) < funcObj(x[j], params.funcType, params.optType)) {
				x[j] = xPos[j];
				vel[j] = velPos[j];
				A_loudness[j] = params.alpha * A_loudness[j];
				r_pulseRatio[j] = r_pulseRatioIni[j] * (1 - exp(-1 * params.gamma * t));
			}
			fitnessByBat[j] = funcObj(x[j], params.funcType, params.optType);
			bestPos = bestBat(fitnessByBat, params.optType);
			bestSol = x[bestPos];
			bestFit = fitnessByBat[bestPos];
		}
	}
}

int main(int argc, char* argv[])
{
	ConfigParams paramsF1;
	paramsF1.d = 2;
	paramsF1.popSize = 1000;
	paramsF1.numIter = 100;
	paramsF1.alpha = 1.0;
	paramsF1.gamma = 0.9;
	paramsF1.r_ini = 0.5;
	paramsF1.qMin = 0;
	paramsF1.qMax = 2;
	paramsF1.lower = -32.768;
	paramsF1.upper = 32.768;
	paramsF1.funcType = FUNC_ACK;
	paramsF1.optType = OPT_MIN;

	ConfigParams paramsF2;
	paramsF2.d = 2;
	paramsF2.popSize = 1000;
	paramsF2.numIter = 100;
	paramsF2.alpha = 1.0;
	paramsF2.gamma = 0.9;
	paramsF2.r_ini = 0.12;
	paramsF2.qMin = 0;
	paramsF2.qMax = 2;
	paramsF2.lower = -500;
	paramsF2.upper = 500;
	paramsF2.funcType = FUNC_SCH;
	paramsF2.optType = OPT_MIN;

	ConfigParams paramsF3;
	paramsF3.d = 2;
	paramsF3.popSize = 1000;
	paramsF3.numIter = 100;
	paramsF3.alpha = 1.0;
	paramsF3.gamma = 0.9;
	paramsF3.r_ini = 0.5;
	paramsF3.qMin = 0;
	paramsF3.qMax = 2;
	paramsF3.lower = -100;
	paramsF3.upper = 100;
	paramsF3.funcType = FUNC_BE3;
	paramsF3.optType = OPT_MAX;

	vector<double> bestSol;
	double bestFit;
	BatFunction(paramsF1, bestSol, bestFit);

	cout << "MEJOR SOLUCION:\n";
	for (int i = 0; i < bestSol.size(); i++) {
		cout << bestSol[i] << endl;
	}
	cout << "\nValor f(x) = " << bestFit << endl;

	char a;
	cin >> a;
	return 0;
}