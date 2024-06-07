
//this creates the linear operator in fourier-space


#include <armadillo>
#include <cufftw.h>
#include "const.h"
#include <cmath>
#include <complex>
using namespace std;
using namespace arma;

void linear(cx_mat & L){
    /*
		Routine to compute the linear operator in the Navier-Stokes equations. This consists of the generalized form of the linear (hyper)-friction and viscous hyper-viscosity. 

    */
    
	for(int j = 0; j < Ny/2 ; j++){
		for(int i = 0; i < Nxf; i++){
			L(i,j) = complex<double>( -nu * pow(pow((2.0 * pi * double(i) / Lx),2.0) +pow((2.0 * pi * double(j) / Ly),2.0),nupower) - alpha*pow(pow((2.0 * pi * double(i) / Lx),2.0) +pow((2.0 * pi * double(j) / Ly),2.0),alphapower) ,  0.0);

			L(i,Ny-j-1) = complex<double>( -nu * pow(pow((2.0 * pi * double(i) / Lx),2.0) +pow((2.0 * pi * double(-j-1) / Ly),2.0),nupower) - alpha*pow(pow((2.0 * pi * double(i) / Lx),2.0) +pow((2.0 * pi * double(-j-1) / Ly),2.0),alphapower) ,  0.0);
		}
	}

	L(0,0) = complex<double>(0.0,0.0);

    return;
}




void setup_ETDRK(cx_mat L_E,cx_mat & Q1,cx_mat &Q2,cx_mat & Q3,cx_mat & Q4,cx_mat & Q5,cx_mat &F1, cx_mat & F2, cx_mat & F3, cx_mat &E1, cx_mat& E2, double h){

/*
Routine to set up commonly used operators for ETDRK4 timestepping routine
*/

cx_mat LR(Nxf,Ny,fill::zeros);
int M = 32;									//number of complex contour points


/*
This is the code for the original ETDRK4 from Cox and Matthews and also ETDRK-B (same)
*/


E1 = exp(h*L_E);
E2 = exp(0.5*h*L_E);


if(FLAG_ETDRK_ORDER == 2){
	
	for(int i  = 0; i < M; i++){

		LR = h*L_E + exp(complex<double>(0.0,1.0)*2.0*pi*(double(i)-0.5)/ double(M));
		Q1 += (exp(LR) - 1.0) / LR;
		F1 += (exp(LR) - 1.0 - LR) / pow(LR,2.0);

	}

	Q1 *= h / double(M);
	F1 *= h / double(M);

}
else if(FLAG_ETDRK_ORDER == 3){
	
	for(int i  = 0; i < M; i++){

		LR = h*L_E + exp(complex<double>(0.0,1.0)*2.0*pi*(double(i)-0.5)/ double(M));
	
		Q1 += (exp(0.5*LR) - 1.0) / LR;
		Q2 += (exp(LR) - 1.0) / LR; 
		F1 += (-4.0 - LR + exp(LR)%(4.0-3.0*LR+pow(LR,2.0))) / pow(LR,3.0);
		F2 += (2.0 + LR + exp(LR)%(-2.0 + LR)) / pow(LR,3.0);
		F3 += (-4.0 - 3.0*LR - pow(LR,2.0) + exp(LR)%(4.0-LR)) / pow(LR,3.0);
	}

	Q1 *= h / double(M);
	Q2 *= h / double(M);
	F1 *= h / double(M);
	F2 *= h / double(M);
	F3 *= h / double(M);
}
else if(FLAG_ETDRK_ORDER == 4){
for(int i  = 0; i < M; i++){

	LR = h*L_E + exp(complex<double>(0.0,1.0)*2.0*pi*(double(i)-0.5)/ double(M));
	
	Q1 += (exp(0.5*LR) - 1.0) / LR;
	Q2 += (exp(0.5*LR)%(LR-4.0) + LR + 4.0) / pow(LR,2.0); 
	Q3 += 2.0*(2.0*exp(0.5*LR) - LR - 2.0) / pow(LR,2.0); 
	Q4 += (exp(LR)%(LR-2.0) + LR + 2.0) / pow(LR,2.0); 
	Q5 += 2.0*(exp(LR) - LR - 1.0) / pow(LR,2.0); 
	F1 += (-4.0 - LR + exp(LR)%(4.0-3.*LR+pow(LR,2.0))) / pow(LR,3.0);
	F2 += (2.0 + LR + exp(LR)%(-2.0 + LR)) / pow(LR,3.0);
	F3 += (-4.0 - 3.0*LR - pow(LR,2.0) + exp(LR)%(4.0-LR)) / pow(LR,3.0);
}

Q1 *= h / double(M);
Q2 *= h / double(M);
Q3 *= h / double(M);
Q4 *= h / double(M);
Q5 *= h / double(M);
F1 *= h / double(M);
F2 *= h / double(M);
F3 *= h / double(M);
}
return;
}
