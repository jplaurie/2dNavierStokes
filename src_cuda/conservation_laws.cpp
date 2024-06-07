
// This compute the linear and nonlinear energies, and total wave action


#include <iostream>
#include <cmath>
#include <armadillo>
#include <complex>
#include "const.h"
#include <cufftw.h>
#include <fstream>
#include <iomanip> 

using namespace std;
using namespace arma;

static double k,w2, factor,Energy_tot, Enstrophy_tot, Energy_diss_nu, Energy_diss_alpha, Enstrophy_diss_nu, Enstrophy_diss_alpha;


void conservation_laws(double runtime, int out_count, cx_mat w_hat, double& E_out){

    
    Energy_tot = 0.0;
    Enstrophy_tot=0.0;
    Energy_diss_nu = 0.0;
    Energy_diss_alpha = 0.0;
    Enstrophy_diss_nu = 0.0;
    Enstrophy_diss_alpha = 0.0;
    w2=0.0;
    k=0.0;
    
    for(int j = 0; j < Ny/2; j++){
        for(int i = 0; i < Nxf; i++){
        
           	factor = 2.0;
            if(i==0 || i == Nxf-1) factor = 1.0;

            k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(-j-1)/ Ly,2.0) , 0.5);
           
            w2 = pow( abs(w_hat(i,Ny-j-1)),2.0);
           
            Energy_tot +=  0.5*factor * w2/ (k*k);
            Enstrophy_tot += 0.5*factor * w2;
            
            Energy_diss_nu += factor * nu * pow(k*k,nupower-1) * w2;
            Enstrophy_diss_nu += factor * nu * pow(k*k,nupower) * w2;
            Energy_diss_alpha += factor * alpha * pow(k*k,alphapower-1) * w2;
            Enstrophy_diss_alpha += factor * alpha * pow(k*k,alphapower) * w2;

            if(j==0 && i==0) continue;
           
            k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(j)/ Ly,2.0) , 0.5); 
            w2 = pow( abs(w_hat(i,j)),2.0);
   
            Energy_tot += 0.5* factor * w2/ (k*k);
            Enstrophy_tot += 0.5*factor * w2;
            
            Energy_diss_nu += factor * nu * pow(k*k,nupower-1.0) * w2;
            Enstrophy_diss_nu += factor * nu * pow(k*k,nupower) * w2;
            Energy_diss_alpha += factor * alpha * pow(k*k,alphapower-1.0) * w2;
            Enstrophy_diss_alpha += factor * alpha * pow(k*k,alphapower) * w2;

        
        }
    }
    E_out = Energy_tot; //Updates E_out for terminal output

   
    ostringstream out_E;
    out_E << "./output/energy." << setw(6) << setfill('0') << out_count << ends;                            //creates file name for outputting ENERGY DENSITY at time slice
    string filenameE = out_E.str();
    ofstream fout_E(filenameE.c_str());
    fout_E << scientific;
    fout_E.precision(12);

        fout_E << runtime << " " << Energy_tot << " " << Enstrophy_tot << endl;
    
    fout_E.close();
    
 
    ostringstream out_Dis;
    out_Dis << "./output/dissipation." << setw(6) << setfill('0') << out_count << ends;                            //creates file name for outputting data at time slice
    string filenameDis = out_Dis.str();
    ofstream fout_Dis(filenameDis.c_str());
    fout_Dis << scientific;
    fout_Dis.precision(12);

    fout_Dis << runtime << " " << Energy_diss_alpha << " " << Energy_diss_nu << " " << Enstrophy_diss_alpha << " " << Enstrophy_diss_nu << endl;
    
    fout_Dis.close();
    
	
	if(FLAG_TIME_SERIES_MODES == true){

	ostringstream out_Modes;
    out_Modes << "./output/modes." << setw(6) << setfill('0') << out_count << ends;                            //creates file name for outputting data at time slice
    string filenameModes = out_Modes.str();
    ofstream fout_Modes(filenameModes.c_str());
    fout_Modes << scientific;
    fout_Modes.precision(12);
    
    fout_Modes << runtime << " " << real(w_hat(1,0)) << " " << imag(w_hat(1,0)) << " " << real(w_hat(0,1)) << " " << imag(w_hat(0,1)) << " " << real(w_hat(1,1)) << " " << imag(w_hat(1,1)) << " " << real(w_hat(2,1)) << " " << imag(w_hat(2,1)) << " " << real(w_hat(0,3)) << " " << imag(w_hat(0,3)) << endl;
    
    fout_Modes.close();
   
   }
    
    return;
}



void output_w( cx_mat & w_hat, mat & w, int out_count,fftw_plan & IFFTN){
    

    fftw_execute(IFFTN);
    
 
    ostringstream out_w;
    out_w << "./data/w." << setw(6) << setfill('0') << out_count << ends;			//creates file name for outputting data at time slice
    string filename = out_w.str();
    ofstream fout_w(filename.c_str());
    fout_w << scientific;
    fout_w.precision(12);
    

	for(int j=0; j < Ny; j++){
        for(int i=0; i < Nx; i++){
      
            fout_w << w(i,j) << " ";
        }
        fout_w << endl;
    }
    
    fout_w.close();

    
}







