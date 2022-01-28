
#include <iostream>
#include <cmath>
#include <armadillo>
#include <complex>
#include "const.h"
#include <fftw3.h>
#include <fstream>
#include <iomanip> 

using namespace std;
using namespace arma;


static int ib = 0;
rowvec Energy_flux(NBIN,fill::zeros), Enstrophy_flux(NBIN,fill::zeros);
static double k_bin,k,factor, fenstrophy,fenergy;


void flux(double runtime, cx_mat w_hat, cx_mat adv_hat, rowvec & Energy_flux_avg, rowvec & Enstrophy_flux_avg, int out_count,int avg_count){
    
    
    k=0.0;
    
    
    Energy_flux.zeros();
    Enstrophy_flux.zeros();

    k_bin = min(2.0*pi / Lx, 2.0*pi/ Ly);
    
    
    for(int j =0; j< Ny/2; j++){
        for(int i =0; i < Nxf; i++){
        
            factor = 2.0;
            if(i==0 || i == Nxf-1) factor=1.0;

  			k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(-j-1)/ Ly,2.0) , 0.5);
            ib = int(k/k_bin);

            fenstrophy = real(w_hat(i,Ny-j-1))*real(adv_hat(i,Ny-j-1)) + imag(w_hat(i,Ny-j-1))*imag(adv_hat(i,Ny-j-1));
            fenergy = fenstrophy/(k*k);
            
            for(int l=0; l <= ib; l++){
                if(ib < NBIN){
                Energy_flux(l) += factor* fenergy;
                Enstrophy_flux(l) += factor * fenstrophy;
                }
            }
             
            if(j==0 && i==0) continue; 

            k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(j)/ Ly,2.0) , 0.5);
            ib = int(k/k_bin);
            
            fenstrophy = real(w_hat(i,j))*real(adv_hat(i,j)) + imag(w_hat(i,j))*imag(adv_hat(i,j));
            fenergy= fenstrophy/(k*k);
            
            for(int l=0; l <= ib; l++){
                if(ib < NBIN){
                Energy_flux(l) += factor* fenergy;
                Enstrophy_flux(l) += factor * fenstrophy;
                }
            }
    
            
        }
    }
  
    
    Energy_flux_avg += Energy_flux;
    Enstrophy_flux_avg += Enstrophy_flux;
    
    
    ostringstream out_flux;
    out_flux << "./output/flux." << setw(6) << setfill('0') << out_count << ends;			//creates file name for outputting data at time slice
    string filename = out_flux.str();
    ofstream fout_flux(filename.c_str());
    fout_flux << scientific;
    fout_flux.precision(12);
    
    for(int i=0; i < NBIN; i++){
        
        fout_flux << (i*k_bin) << " " << Energy_flux(i) << " " << Enstrophy_flux(i) << " " << Energy_flux_avg(i) / double(avg_count) << " " << Enstrophy_flux_avg(i) / double(avg_count) << endl;
    }
    
    fout_flux.close();
    

    
}
