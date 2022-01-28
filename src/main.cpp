/*

Programme to solve the 2D Navier-Stokes equation  w_t = - u.grad w -alpha w -nu* (k^2)^p w 


Author: Jason Laurie
Date: 08/10/2010


*/

#include <iostream>
#include <cmath>
#include <armadillo>
#include <complex>
#include "const.h"
#include "functions.h"
#include <time.h>
#include <fstream>
#include <fftw3.h>
#include <random>
#include <chrono>
#include <omp.h>


using namespace std;
using namespace arma;

int main(){
	
   
	int out_count =0;
 	int avg_count =0;
	double start = 0.0;
	double finish = 0.0;
	double runtime =0.0;
  double E_out = 0.0;
// 	start = omp_get_wtime();
	int nthreads = 0;

	
   	
   	if(num_threads == 0){
   		nthreads = omp_get_max_threads();
   	}
   	else{
   		nthreads = num_threads;
   	}
   	
   	omp_set_num_threads(nthreads);
    
    if(nthreads > 1){
    	int err = fftw_init_threads();
     	if (err==0){
        	cout << "thread creation error: " << err << endl;
     	}
     	else{
        	fftw_plan_with_nthreads(nthreads);
        	cout << "number of parallel threads used by fftw = " << nthreads << endl;
  		}
  	}
  	else{
  		cout << "using a single thread" << endl;
  	}


  start = omp_get_wtime(); 

    
  cx_mat w_hat(Nxf,Ny,fill::zeros),w_hat_temp(Nxf,Ny,fill::zeros), w_hat_M(Mxf,My,fill::zeros), step1(Nxf,Ny,fill::zeros), step2(Nxf,Ny,fill::zeros), step3(Nxf,Ny,fill::zeros), step4(Nxf,Ny,fill::zeros), NL1(Nxf,Ny,fill::zeros), NL2(Nxf,Ny,fill::zeros), NL3(Nxf,Ny,fill::zeros), F1(Nxf,Ny,fill::zeros),F2(Nxf,Ny,fill::zeros),F3(Nxf,Ny,fill::zeros), Q1(Nxf,Ny,fill::zeros),Q2(Nxf,Ny,fill::zeros),Q3(Nxf,Ny,fill::zeros),Q4(Nxf,Ny,fill::zeros),Q5(Nxf,Ny,fill::zeros), E1(Nxf,Ny,fill::zeros),E2(Nxf,Ny,fill::zeros), L(Nxf,Ny,fill::zeros), adv_hat(Nxf,Ny,fill::zeros),noise(Nxf,Ny,fill::zeros);


  	mat w(Nx,Ny,fill::zeros),w_M(Mx,My,fill::zeros),famp(Nxf,Ny,fill::zeros);

    rowvec Energy_flux_avg(NBIN,fill::zeros),Enstrophy_flux_avg(NBIN,fill::zeros),Energy_spec_avg(NBIN,fill::zeros),Enstrophy_spec_avg(NBIN,fill::zeros);
    
  	fftw_plan FFTM;								//declare plans
  	fftw_plan IFFTM;
 	  fftw_plan FFTN;
  	fftw_plan IFFTN;
    
  	FFTN = fftw_plan_dft_r2c_2d(Ny,Nx, (double*) w.memptr(), (fftw_complex*) w_hat.memptr(), FFTW_PATIENT);
  	IFFTN = fftw_plan_dft_c2r_2d(Ny,Nx, (fftw_complex*) w_hat.memptr(), (double*) w.memptr(), FFTW_PATIENT);
  	FFTM = fftw_plan_dft_r2c_2d(My,Mx, (double*) w_M.memptr(), (fftw_complex*) w_hat_M.memptr(), FFTW_PATIENT);
  	IFFTM = fftw_plan_dft_c2r_2d(My,Mx, (fftw_complex*) w_hat_M.memptr(), (double*)w_M.memptr(), FFTW_PATIENT);




  	//set up random forcing
  	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();  
  	mt19937_64 generator(seed1);
  	normal_distribution<double> distribution(0.0,1.0);

    read(w, w_hat, runtime, out_count, FFTN);      //reads in initial data
  

	   system("mkdir ./output");			//Creates the Folders for Outputting	
	
  	linear(L);     //Creates linear operator
   	
   	if(FLAG_TIMESTEP_ETDRK == true){
   		setup_ETDRK(L,Q1,Q2,Q3,Q4,Q5,F1,F2,F3,E1,E2,dt);
   	}

if(FLAG_FORCING_ON == true){

      if(FLAG_FORCE_EXP == true){
          forcing_exponential(famp,kf,dk);
           
      }
      else if(FLAG_FORCE_STATIC == true){
        

        forcing_grid(famp,kf,dk);
    

      }
      else{
        
          forcing_annulus(famp,kf,dk); //Initialise forcing
      }
    
  }


  record_parameters();


    //==============Time stepping routine=====================================================
    
   // start = omp_get_wtime();
    
  	for( int time_step = 1; time_step <= nsteps; time_step++ ){
       
    	runtime += dt;
        

    	if(FLAG_FORCING_ON == true){
    
    		 /*  This is the section of code that produces the forcing using normal distributed real and imaginary components at a given wave number*/

        noise.zeros();
        if(FLAG_FORCE_STATIC == true){
          noise = famp*pow(0.5,0.5)*complex<double>(1.0,0.0);
        
        symmetry(noise);
        }
        else{


        
        for(int j = 0; j < Ny; j++){
      		for(int i = 0; i < Nxf; i++){
          			if(famp(i,j) > 0.0){  
            			noise(i,j) = famp(i,j)*pow(0.5,0.5)*complex<double>(distribution(generator),distribution(generator));          			
          			
                }
        		}
      		}
    
      		symmetry(noise);
      	}
      }

      //	start = omp_get_wtime();

        if(FLAG_TIMESTEP_ETDRK == true){
          
        	w_hat += pow(dt,0.5)*noise; // euler noise
    	 

      //=========================ETDRK4======================================
/*
			NL1 = NL(w_hat,0.0, L , w_M, w_hat_M, FFTM, IFFTM);
    		step1 = E2*w_hat + Q1*NL1;
    		NL2 = NL(step1,0.5*dt, L , w_M, w_hat_M, FFTM, IFFTM); 
    		step2 = E2*w_hat + Q1*NL2;
    		NL3 = NL(step2,0.5*dt, L , w_M, w_hat_M, FFTM, IFFTM);
    		step3 = E2*step1 + Q1*(2.0*NL3 -NL1);
    		step4 = E1*w_hat + F1*NL1 + 2.0*F2*(NL2+NL3) + F3*NL(step3,dt, L , w_M, w_hat_M, FFTM, IFFTM);
    		w_hat = step4;
    	//==============================================================================================
*/    

      if(FLAG_ETDRK_ORDER == 2){
        NL1 = nonlinear(w_hat,0.0, L , w_M, w_hat_M, FFTM, IFFTM);
        step1 = E1%w_hat + Q1%NL1;
        NL2 = nonlinear(step1,dt, L , w_M, w_hat_M, FFTM, IFFTM); 
        step2 = step1 +  F1 % (NL2 - NL1);
        w_hat = step2;

      }
      else if(FLAG_ETDRK_ORDER == 3){
        NL1 = nonlinear(w_hat,0.0, L , w_M, w_hat_M, FFTM, IFFTM);
        step1 = E2%w_hat + Q1%NL1;
        NL2 = nonlinear(step1,0.5*dt, L , w_M, w_hat_M, FFTM, IFFTM); 
        step2 = E1%w_hat + Q2%(2.0*NL2-NL1);
        NL3 = nonlinear(step2,dt, L , w_M, w_hat_M, FFTM, IFFTM);
        step3 = E1 % w_hat + F1 % NL1 + 4.0* F2 % NL2 + F3 % NL3;
        w_hat = step3;
      }
      else if(FLAG_ETDRK_ORDER == 4){
    	//=========================ETDRK4-B======================================

        NL1 = nonlinear(w_hat,0.0, L , w_M, w_hat_M, FFTM, IFFTM);
    		step1 = E2%w_hat + Q1%NL1;
    		NL2 = nonlinear(step1,0.5*dt, L , w_M, w_hat_M, FFTM, IFFTM); 
    		step2 = E2%w_hat + Q2%NL1 + Q3%NL2;
    		NL3 = nonlinear(step2,0.5*dt, L , w_M, w_hat_M, FFTM, IFFTM);
    		step3 = E1 % w_hat + Q4 % NL1 + Q5%NL3;
    		step4 = E1 % w_hat + F1 % NL1 + 2.0*F2%(NL2+NL3) + F3%nonlinear(step3,dt, L , w_M, w_hat_M, FFTM, IFFTM);
    		w_hat = step4;
      }
    	//============================================================================================== 
		}
    	else{

    		step1 = dt*nonlinear(w_hat,0.0, L , w_M, w_hat_M, FFTM, IFFTM);
    		step2 = dt*nonlinear( w_hat + step1 + pow(dt,0.5)*noise,dt, L , w_M, w_hat_M, FFTM, IFFTM);
    		w_hat += 0.5*(step1 + step2) + pow(dt,0.5)*noise;
        
    		w_hat %= exp(dt*L);  //transforms data to u=exp(Ldt)v  
         
         

    	//=====================================================================
    	}

    //	finish = omp_get_wtime();

    //	cout << "time_stepping = " << finish-start << endl;
    	if( (time_step % outstep) == 0 ){
        
            
      		out_count++;
      		avg_count++;
            
      		
            
         	w_hat_temp = w_hat;  
      		
      		spectrum(w_hat, Energy_spec_avg, Enstrophy_spec_avg, out_count,avg_count); //computes spectrum    
      		conservation_laws(runtime,out_count,w_hat,E_out);       //computes energy 
      		adv_hat = nonlinear(w_hat,0.0, L , w_M, w_hat_M, FFTM, IFFTM);     
         	flux(runtime,w_hat,adv_hat,Energy_flux_avg,Enstrophy_flux_avg,out_count,avg_count); //computes flux        
      		output_w(w_hat,w,out_count,IFFTN);
        
        	w_hat = w_hat_temp;

        	cout << "time = " << runtime << " file = " << out_count << " Energy = " << E_out << endl;        //outputs runtime to screen
      		ofstream fout_count("./data/curframe.dat");
      		fout_count << scientific;
      		fout_count.precision(12);
      		fout_count << runtime << " " << out_count << endl;
    	}
       
  	}
    
	 
	fftw_destroy_plan(FFTM);
	fftw_destroy_plan(IFFTM);			//destroy plans
	fftw_destroy_plan(FFTN);
	fftw_destroy_plan(IFFTN);

	/*time (&end);			//end clock

	elapsed_time = difftime(end,start);	//work out time of code
	cout << "time taken for code is = " << elapsed_time << endl;	//output time
*/
    finish = omp_get_wtime();                       //end clock
    cout << "time taken for code is = " << finish-start << endl;    //output time
  

	return 0;
	
}
	

