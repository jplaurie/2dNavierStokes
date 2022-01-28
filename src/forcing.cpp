

#include <iostream>
#include <cmath>
#include <blitz/array.h>
#include <complex>
#include "Const.h"
#include <fftw3.h>
#include <fstream>

using namespace std;
using namespace blitz;

double sdt = pow(dt,0.5);
double phase = 0.0;


void symmetryf(Array<complex<double>, 2> A){

    A(0,0) = complex<double>(0.0,0.0); //forces the zeroth wave number to be zero
    A(Nx/2,0) = complex<double>(real(A(Nx/2,0)),0.0);
    A(Nx/2,Nyf-1) = complex<double>(real(A(Nx/2,Nyf-1)),0.0);
    A(0,Nyf-1) = complex<double>(real(A(0,Nyf-1)),0.0);
     
    for(int i = 1; i< Nx/2; i++){
        A(Nx-i,0) = conj(A(i,0));              // makes sure that w_hat(-kx, 0 ) = w_hat(kx,0)
        A(Nx-i,Nyf-1) = conj(A(i,Nyf-1));      // makes sure that w_hat(-kx,-Ny/2) = w_hat(kx, -Ny/2) 
     }


}


void forcing(blitz::Array<complex<double>,2> w_hat,blitz::Array<double,2> famp){
    /*
        Subroutine to 
    */
    
    for(int i =0 ; i < Nx; i++){
        for(int j =0; j < Ny/2 + 1; j++){
            phase = 2.0*pi*  double( rand() ) /double( RAND_MAX + 1.0 );    
            w_hat(i,j) += sdt * famp(i,j)*complex<double>(cos(phase),sin(phase));
        }
    }
      
    symmetryf(w_hat);
    
    
    return;
}
