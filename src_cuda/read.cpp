
//this code reads in the initial data from file ./Initial.dat

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


void read(mat & w,cx_mat & w_hat, double& runtime, int& out_count, fftw_plan FFTN){
    /*
        Reads in initial condition from file number given by curframe.dat

    */

    
    fstream filein("./data/curframe.dat");
    filein.precision(12);
    filein >> runtime >> out_count;
    

    if(out_count < 0){
    /*	for(int j=0; j < Ny; j++){
            for(int i=0; i < Nx; i++){
      
            	w(i,j) = cos( (2*pi*i/Nx) + (4.0*pi*j/Ny) );
        	}
    	}*/
       	
    	out_count = 0;

	ofstream fout_read("./data/w.000000");
    	fout_read.precision(12);

    	for(int j=0; j < Ny; j++){
            for(int i=0; i < Nx; i++){
            	fout_read << w(i,j) << " ";
        	}
        fout_read << endl;
    	}
    	fout_read.close();


    	cout << "out_count < 0" << endl; 
    	cout << "Simulation starting...from zero state" << endl;
    	cout << "time = " << runtime << endl;
    	cout << "out_count = 0" << endl; 
    }
    else{
    	ostringstream in_data;
    	in_data << "./data/w." << std::setw(6) << std::setfill('0') << out_count << ends;
    	string filename = in_data.str();
    	ifstream filein2(filename.c_str());
    	filein2.precision(12);
   
    	for(int j = 0; j < Ny; j++){
        	for(int i = 0; i < Nx; i++){
            	filein2 >> w(i,j);
        	}
    	}

    	cout << "Simulation starting..." << endl;
    	cout << "time = " << runtime << endl;
    	cout << "Starting frame = w." << std::setw(6) << std::setfill('0') << out_count << endl; 


    }



    fftw_execute(FFTN);
    
    w_hat /= double(Nx*Ny);
    
    return;
}


void record_parameters(){


    cout << "FLAG_FORCING_ON = " << FLAG_FORCING_ON << endl;
    cout << "FLAG_FORCE_EXP = " << FLAG_FORCE_EXP << endl;
    cout << "FLAG_FORCE_AMP_RESCALE = " << FLAG_FORCE_AMP_RESCALE << endl;
    cout << "FLAG_QG = " << FLAG_QG << endl;
    cout << "FLAG_TIMESTEP_ETDRK = " << FLAG_TIMESTEP_ETDRK << endl;
    cout << "FLAG_ETDRK_ORDER = " << FLAG_ETDRK_ORDER << endl;
    cout << endl;
    cout << "Nx = " << Nx << " Ny = " << Ny << " Lx = " << Lx << " Ly = " << Ly << endl;
    cout << "Aspect ratio = " << asp_rat << endl;
    cout << "dt = " << dt << endl;
    cout << "total number of timesteps = " << nsteps << endl;
    cout << "output every " << outstep << " timesteps" << endl;
    cout << "alpha = " << alpha << " alpha_power = " << alphapower << " nu = " << nu << " nu_power = " << nupower << endl;
    cout << "kf = " << kf << " dk = " << dk << " Amp = " << Amp << endl;  
    if(FLAG_QG == true){
        cout << "beta = " << beta << endl;
    }
    ofstream fout_data("./parameters.txt");
    fout_data << "FLAG_FORCING_ON = " << FLAG_FORCING_ON << endl;
    fout_data << "FLAG_FORCE_EXP = " << FLAG_FORCE_EXP << endl;
    fout_data << "FLAG_FORCE_AMP_RESCALE = " << FLAG_FORCE_AMP_RESCALE << endl;
    fout_data << "FLAG_QG = " << FLAG_QG << endl;
    fout_data << "FLAG_TIMESTEP_ETDRK = " << FLAG_TIMESTEP_ETDRK << endl;
    fout_data << "FLAG_ETDRK_ORDER = " << FLAG_ETDRK_ORDER << endl;
    fout_data << endl;
    fout_data << "Nx = " << Nx << " Ny = " << Ny << " Lx = " << Lx << " Ly = " << Ly << endl;
    fout_data << "aspect ratio = " << asp_rat << endl;
    fout_data << "dt = " << dt << endl;
    fout_data << "total number of timesteps = " << nsteps << endl;
    fout_data << "output every " << outstep << " timesteps" << endl;
    fout_data << "alpha = " << alpha << " alpha_power = " << alphapower << " nu = " << nu << " nu_power = " << nupower << endl;
    fout_data << "kf = " << kf << " dk = " << dk << " Amp = " << Amp << endl;
    if(FLAG_QG == true){
        fout_data << "beta = " << beta << endl;
    }
    fout_data.close();






    return;
}
