

#include <iostream>
#include <cmath>
#include <armadillo>
#include <complex>
#include "const.h"
#include <fftw3.h>
#include <fstream>
#include <omp.h>
#include <iomanip> 

using namespace std;
using namespace arma;

static int ib = 0;
rowvec Energy_spec(NBIN,fill::zeros), Enstrophy_spec(NBIN,fill::zeros);
static double k_bin,k,factor,w2;


void spectrum( cx_mat A,rowvec & Energy_spec_avg, rowvec & Enstrophy_spec_avg,int out_count,int avg_count){
    
    k=0.0;
    w2=0.0;
    Energy_spec.zeros();
    Enstrophy_spec.zeros();

    k_bin = min(2.0*pi / Lx, 2.0*pi/ Ly);
    
  
    for(int j =0; j< Ny/2; j++){
        for(int i =0; i < Nxf; i++){
           
            factor = 1.0;
            if(i==0 || i== Nxf-1) factor = 0.5;

             k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(-j-1)/ Ly,2.0) , 0.5);
            ib = int(k/k_bin);
            w2 = pow( abs(A(i,Ny-j-1)) , 2.0);
            Enstrophy_spec(ib) += factor*w2;
            Energy_spec(ib) += factor*w2/(k*k);
                 
            k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(j)/ Ly,2.0) , 0.5);
            ib = int(k/k_bin);
            w2 = pow( abs(A(i,j)) , 2.0);
            if(i==0 && j==0) k=1.0;
            Enstrophy_spec(ib) += factor*w2;
            Energy_spec(ib) += factor*w2/(k*k);
   
        }
    }
    
    Enstrophy_spec_avg += Enstrophy_spec;
    Energy_spec_avg += Energy_spec;

    ostringstream out_spec;
    out_spec << "./output/spec." << setw(6) << setfill('0') << out_count << ends;			//creates file name for outputting data at time slice
    string filename = out_spec.str();
    ofstream fout_spec(filename.c_str());
    fout_spec << scientific;
    fout_spec.precision(12);
    
    for(int i=0; i < NBIN; i++){
            
            fout_spec << (i*k_bin) << " " << Energy_spec(i) << " " << Enstrophy_spec(i) << " " << Energy_spec_avg(i)/double(avg_count) << " " << Enstrophy_spec_avg(i) / double(avg_count) <<  endl;
    }
    
    fout_spec.close();
    
    
    
    return;
}