#include <armadillo>
#include <cufftw.h>
#include "const.h"
#include <complex>
#include <cmath>
//#include <omp.h>


using namespace std;
using namespace arma;


static cx_mat  w_hat_N(Nxf,Ny,fill::zeros);
static mat R(Mx,My,fill::zeros),U(Mx,My,fill::zeros);

void symmetry(cx_mat &A){
/*
	function that sets the required symmetries at the boundary wave numbers.
*/
	A(0,0) = complex<double>(0.0,0.0); //forces the zeroth wave number to be zero
    A(0,Ny/2) = complex<double>(real(A(0,Ny/2)),0.0);
    A(Nxf-1,Ny/2) = complex<double>(real(A(Nxf-1,Ny/2)),0.0);
    A(Nxf-1,0) = complex<double>(real(A(Nxf-1,0)),0.0);
    
 
    
	for(int j = 1; j < Ny/2; j++){
    	A(0,Ny-j) = conj(A(0,j));				// makes sure that w_hat(0,-ky ) = w_hat^*(0,ky)
    	A(Nxf-1,Ny-j) = conj(A(Nxf-1,j));      // makes sure that w_hat(-Nx/2,-ky) = w_hat^*(-Nx/2,ky) 
    }


}


//=============================================================================================================
void PutArray(cx_mat & X_M, cx_mat X_N){		

	/*
	Maps the standard size array into the larger Array_M size to perform 3/2 dealiasing rule
	*/
    
	X_M.zeros();

	for(int j = 0; j < Ny/2; j++){
		for(int i = 0; i < Nxf; i++){
			X_M(i,j) = X_N(i,j);
			X_M(i,My-j-1) = X_N(i,Ny-j-1);
		}
	}

return;

}

//================================================================================================================
void GetArray(cx_mat & X_N,cx_mat X_M){ 
	/*
	Maps Array_M back to standard size array and then implements the required symmetry conditions at the boundary wave numbers
	*/


	X_N.zeros();
			
	for(int j = 0; j < Ny/2; j++){
		for(int i = 0; i < Nxf; i++){
			X_N(i,j) = X_M(i,j);
			X_N(i,Ny-j-1) = X_M(i,My-j-1);
		}
	}


    symmetry(X_N);

    return;
}


//================================================================================================================
void d_dy(cx_mat & X_M){	
	/* 
		Calculates the y derivative using static array
    */

	for(int j = 0; j < Ny/2; j++){
    	for(int i = 0; i < Nxf; i++){
			X_M(i,j) *= complex<double>(0.0 , (2.0 * pi * double(j) / Ly));
			X_M(i,My-j-1) *= complex<double>(0.0, (2.0 * pi * double(-j-1) / Ly));
		
		}
	}



    
}

void d_dx(cx_mat & X_M){
	/*
		calculates the x derivative using static array
	*/
	
	for(int j = 0; j < Ny/2; j++){
		for(int i = 0; i < Nxf; i++){
			X_M(i,j) *= complex<double>(0.0 , (2.0 * pi * double(i) / Lx));
			X_M(i,My-j-1) *= complex<double>(0.0, (2.0 * pi * double(i) / Lx));
		
		}
	}

	
}


void invlap(cx_mat & X_M){
	/*
	Computes the inverse Laplacian
	*/
	
	for(int j = 0; j < Ny/2; j++){
		for(int i = 0; i < Nxf; i++){
			X_M(i,j) /= -(pow( 2.0 * pi * double(i) / Lx , 2.0) + pow( 2.0 * pi * double(j) / Ly , 2.0));
			X_M(i,My-j-1) /= -(pow( 2.0 * pi * double(i) / Lx , 2.0) + pow(2.0 * pi * double(-j-1) / Ly , 2.0));
		}
	}

	X_M(0,0) = complex<double>(0.0,0.0);
	
}





cx_mat nonlinear(  cx_mat A, double da, cx_mat L, mat & w_M,cx_mat & w_hat_M, fftw_plan FFTM, fftw_plan IFFTM){

    w_hat_N.zeros();
 	w_hat_N = A;

 
	if(FLAG_TIMESTEP_ETDRK == false){
   		w_hat_N %= exp(L*da);  //transforms data into u=exp(L*dt)*v
	}
    
    R.zeros();				//initialize R
	U.zeros();

	PutArray(w_hat_M,w_hat_N);		//puts w_hat into larger array
	 
	d_dx(w_hat_M);					//calculates x derivative
	
	fftw_execute(IFFTM);		//inverse FFT into array U - this is now dw / dx

	R=w_M;

    PutArray(w_hat_M,w_hat_N);			//puts w_hat into larger array
	
    invlap(w_hat_M);                   // computes w - > psi
	 
    d_dy(w_hat_M);						//calculates dpsi/dy derivative
	 
	fftw_execute(IFFTM);		//inverse FFT into O - this is now dpsi/dy
	
	
	R %= w_M;			// R =  dPsi / dy  * dw/dx
	
	PutArray(w_hat_M,w_hat_N);		//puts w_hat into larger array
	
	d_dy(w_hat_M);						//caluclates w derivative
    
	
	fftw_execute(IFFTM);	//inverse FFT into U - this is now dw/dy
	
    U = w_M;
    
	PutArray(w_hat_M,w_hat_N);					//puts w_hat into larger array
	
    invlap(w_hat_M);                               // w -> psi
	
    d_dx(w_hat_M);									//calcualtes dpsi/dx derivative

	fftw_execute(IFFTM)	;	//inverse FFT into O - this is now dpsi/dx
	
	R -= U%w_M;								// R is now  R =   dPsi / dy  * dw/dx - dPsi / dx  * dw/dy

	if(FLAG_QG == true){
		R -= beta * w_M;
	}

    w_M = R;
	
    fftw_execute(FFTM);				//FFT into Fourier space
	
    w_hat_M /= double(Mx*My);
    
	GetArray(w_hat_N,w_hat_M);							//Puts array into smaller (normal) size
    
	symmetry(w_hat_N);
  
    if(FLAG_TIMESTEP_ETDRK == false){
  	  w_hat_N %= exp(-L*da);            //transforms back to v=exp(-L*dt)u
  	}
  	
    return w_hat_N;                   //returns nonlinear term
}





    
