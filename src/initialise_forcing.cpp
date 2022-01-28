#include <iostream>
#include <cmath>
#include <armadillo>
#include <complex>
#include "const.h"
#include <fftw3.h>
#include <fstream>

using namespace std;
using namespace arma;

static double k,kx,ky,energy_inj, enstrophy_inj, factor;
static int forcing_modes;


void forcing_annulus(mat &famp, double kf, double dk){
    
    
    forcing_modes =0;
    energy_inj=0.0;
    enstrophy_inj=0.0;
    
    for(int j =0; j < Ny/2; j++){
    	for(int i =0; i< Nxf; i++){
        
        	factor = 1.0;
       		if( i==0 || i == Nx/2 ) factor=0.5;
        
       		k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(-j-1)/ Ly,2.0) , 0.5);
            
            if( abs(k-kf) <  dk){
                famp(i,Ny-j-1) = Amp;
                enstrophy_inj += factor*pow(famp(i,Ny-j-1),2.0);
                energy_inj += factor*pow(famp(i,Ny-j-1),2.0)/(k*k);
                forcing_modes += (int) 2.0*factor;
            }

            if(i==0 && j ==0) continue;
                
            k = pow( pow(   2.0*pi * double(i) / Lx,2.0)  + pow(   2.0*pi * double(j)/ Ly,2.0) , 0.5);
            
            if( abs(k-kf) < dk){
                famp(i,j) = Amp;
                enstrophy_inj += factor*pow(famp(i,j),2.0);
                energy_inj += factor*pow(famp(i,j),2.0)/(k*k);
                forcing_modes += (int) 2.0*factor;
            }
    
        }
    }
    
    /*
        this makes sure f(-kx,0) = f(kx,0)^* and f(-kx,Ny/2) = f(kx,Ny/2)^*

    */
    for(int j = 1; j < Ny/2; j++){	
        famp(0,Ny-j) = famp(0,j);
        famp(Nxf-1,Ny-j) = famp(Nxf-1,j);
    }
 
   	if(FLAG_FORCE_AMP_RESCALE == true){
    	famp *= sqrt(2.0*alpha/energy_inj);
    	enstrophy_inj *= (2.0*alpha/energy_inj);
    	energy_inj = 2.0*alpha;
    }
    
    cout << "number of forcing modes = " << forcing_modes << endl;
    cout << "energy injection rate = " << energy_inj << endl;
    cout << "enstrophy injection rate = " << enstrophy_inj << endl;
    	
    ofstream fout_force_data("./output/force_values.txt");
    fout_force_data.precision(12);

    fout_force_data << "number of forcing modes = " << forcing_modes << endl;
    fout_force_data << "energy injection rate = " << energy_inj << endl;
    fout_force_data << "enstrophy injection rate = " << enstrophy_inj << endl;
    fout_force_data << "forcing mode kf = " << kf << endl;
    fout_force_data << "forcing width dk = " << dk << endl;


    fout_force_data.close();
    
    ofstream fout_Amp("./output/force_spectrum.dat");
    fout_Amp << scientific;
    fout_Amp.precision(12);


    for(int i = 1; i < Nxf; i++){
    
        for(int j =0; j< Ny/2; j++){
            
            kx = 2.0*pi * double(i-Nxf)/ Lx;
            ky = 2.0*pi * double(j-(Ny/2))/ Ly;
    
    fout_Amp << kx << " " << ky << " " << famp(Nxf-i,j) << endl;
    
        }
    
        for(int j =0; j< Ny/2; j++){
            
            kx = 2.0 * pi * double(i-Nxf+1) / Lx;
            ky = 2.0 * pi * double(j) / Ly;
            
            fout_Amp << kx << " " << ky << " " << famp(Nxf-i,j+(Ny/2)) << endl;
            
        }
        fout_Amp << endl;
        
    }
    
     for(int i =0; i < Nxf-1; i++){
    
         for(int j =0; j< Ny/2; j++){
             
             kx = 2.0*pi * double(i)/ Lx;
             ky = 2.0*pi * double(j-(Ny/2))/ Ly;
             
             fout_Amp << kx << " " << ky << " " << famp(i,j+(Ny/2)) << endl;
             
         }
         
         for(int j =0; j< Ny/2; j++){
             
             kx = 2.0*pi * double(i)/ Lx;
             ky = 2.0*pi * double(j)/ Ly;
             
             fout_Amp << kx << " " << ky << " " << famp(i,j) << endl;
             
         }
         fout_Amp << endl;
     }
          
         
    fout_Amp.close();

 
    
    
    
    
    return;
}

//==============================================================================================================

//==============================================================================================================

//==============================================================================================================

//==============================================================================================================

//==============================================================================================================

//==============================================================================================================

//==============================================================================================================






void forcing_exponential(mat &famp, double kf, double dk){
    
    
    forcing_modes =0;
    energy_inj=0.0;
    enstrophy_inj=0.0;
    
    for(int i =0; i< Nxf; i++){
        
        factor = 1.0;
        if( (i==0) || (i== Nx/2) ) factor=0.5;
        
        for(int j =0; j< Ny/2; j++){
            if(i==0 && j ==0) continue;
                
            k = pow( pow(   2.0*pi * double(i) / Lx,2.0)  + pow(   2.0*pi * double(j)/ Ly,2.0) , 0.5);
            
            famp(i,j) = pow(k/kf,force_power)*exp(-pow(k/kf,force_power));
            enstrophy_inj += factor*pow(famp(i,j),2.0);
            energy_inj += factor*pow(famp(i,j),2.0)/(k*k);
            forcing_modes += (int) 2.0*factor;
            
          
            k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(-j-1)/ Ly,2.0) , 0.5); 
                
            famp(i,Ny-j-1) = pow(k/kf,force_power)*exp(-pow(k/kf,force_power));
            enstrophy_inj += factor*pow(famp(i,Ny-j-1),2.0);
            energy_inj += factor*pow(famp(i,Ny-j-1),2.0)/(k*k);
            forcing_modes += (int) 2.0*factor;
        }
    
    }
    
    
    /*
        this makes sure f(-kx,0) = f(kx,0)^* and f(-kx,Ny/2) = f(kx,Ny/2)^*

    */
    for(int j = 1; j< Ny/2; j++){   
        famp(0,Ny-j) = famp(0,j);
        famp(Nxf-1,Ny-j) = famp(Nxf-1,j);
    }
 
 	if(FLAG_FORCE_AMP_RESCALE == true){
    	famp *= sqrt(2.0*alpha/energy_inj);
    	enstrophy_inj *= (2.0*alpha/energy_inj);
    	energy_inj = 2.0*alpha;
    }

    cout << "number of forcing modes = " << forcing_modes << endl;
    cout << "energy injection rate = " << energy_inj << endl;
    cout << "enstrophy injection rate = " << enstrophy_inj << endl;
        
    ofstream fout_force_data("./output/force_values.txt");
    fout_force_data.precision(12);

    fout_force_data << "number of forcing modes = " << forcing_modes << endl;
    fout_force_data << "energy injection rate = " << energy_inj << endl;
    fout_force_data << "enstrophy injection rate = " << enstrophy_inj << endl;
    fout_force_data << "forcing mode kf = " << kf << endl;
    fout_force_data << "forcing width dk = " << dk << endl;

    fout_force_data.close();


    ofstream fout_Amp("./output/force_spectrum.dat");
    fout_Amp << scientific;
    fout_Amp.precision(12);


 for(int i = 1; i < Nxf; i++){
    
        for(int j =0; j< Ny/2; j++){
            
            kx = 2.0*pi * double(i-Nxf)/ Lx;
            ky = 2.0*pi * double(j-(Ny/2))/ Ly;
    
    fout_Amp << kx << " " << ky << " " << famp(Nxf-i,j) << endl;
    
        }
    
        for(int j =0; j< Ny/2; j++){
            
            kx = 2.0 * pi * double(i-Nxf+1) / Lx;
            ky = 2.0 * pi * double(j) / Ly;
            
            fout_Amp << kx << " " << ky << " " << famp(Nxf-i,j+(Ny/2)) << endl;
            
        }
        fout_Amp << endl;
        
    }
    
     for(int i =0; i < Nxf-1; i++){
    
         for(int j =0; j< Ny/2; j++){
             
             kx = 2.0*pi * double(i)/ Lx;
             ky = 2.0*pi * double(j-(Ny/2))/ Ly;
             
             fout_Amp << kx << " " << ky << " " << famp(i,j+(Ny/2)) << endl;
             
         }
         
         for(int j =0; j< Ny/2; j++){
             
             kx = 2.0*pi * double(i)/ Lx;
             ky = 2.0*pi * double(j)/ Ly;
             
             fout_Amp << kx << " " << ky << " " << famp(i,j) << endl;
             
         }
         fout_Amp << endl;
     }
          
         
    fout_Amp.close();

    return;
}



void forcing_grid(mat &famp, double kf, double dk){
    
    forcing_modes =0;
    energy_inj=0.0;
    enstrophy_inj=0.0;
    
    for(int j =0; j < Ny/2; j++){
    	for(int i =0; i< Nxf; i++){
        
        	factor = 1.0;
       		if( i==0 || i == Nx/2 ) factor=0.5;
        
       		k = pow( pow(   2.0*pi * double(i)/ Lx,2.0)  + pow(   2.0*pi * double(-j-1)/ Ly,2.0) , 0.5);
            
            if( i==int(kf) && j==int(kf)){
                famp(i,Ny-j-1) = Amp;
                enstrophy_inj += factor*pow(famp(i,Ny-j-1),2.0);
                energy_inj += factor*pow(famp(i,Ny-j-1),2.0)/(k*k);
                forcing_modes += (int) 2.0*factor;
            }

            if(i==0 && j ==0) continue;
                
            k = pow( pow(   2.0*pi * double(i) / Lx,2.0)  + pow(   2.0*pi * double(j)/ Ly,2.0) , 0.5);
            
            if(  i==int(kf) && j==int(kf)){
                famp(i,j) = -Amp;
                enstrophy_inj += factor*pow(famp(i,j),2.0);
                energy_inj += factor*pow(famp(i,j),2.0)/(k*k);
                forcing_modes += (int) 2.0*factor;
            }
    
        }
    }
    
    /*
        this makes sure f(-kx,0) = f(kx,0)^* and f(-kx,Ny/2) = f(kx,Ny/2)^*

    */
    for(int j = 1; j < Ny/2; j++){	
        famp(0,Ny-j) = famp(0,j);
        famp(Nxf-1,Ny-j) = famp(Nxf-1,j);
    }
 
   	if(FLAG_FORCE_AMP_RESCALE == true){
    	famp *= sqrt(2.0*alpha/energy_inj);
    	enstrophy_inj *= (2.0*alpha/energy_inj);
    	energy_inj = 2.0*alpha;
    }
    
    cout << "number of forcing modes = " << forcing_modes << endl;
    cout << "energy injection rate = " << energy_inj << endl;
    cout << "enstrophy injection rate = " << enstrophy_inj << endl;
    	
    ofstream fout_force_data("./output/force_values.txt");
    fout_force_data.precision(12);

    fout_force_data << "number of forcing modes = " << forcing_modes << endl;
    fout_force_data << "energy injection rate = " << energy_inj << endl;
    fout_force_data << "enstrophy injection rate = " << enstrophy_inj << endl;
    fout_force_data << "forcing mode kf = " << kf << endl;
    fout_force_data << "forcing width dk = " << dk << endl;


    fout_force_data.close();
    
    ofstream fout_Amp("./output/force_spectrum.dat");
    fout_Amp << scientific;
    fout_Amp.precision(12);


    for(int i = 1; i < Nxf; i++){
    
        for(int j =0; j< Ny/2; j++){
            
            kx = 2.0*pi * double(i-Nxf)/ Lx;
            ky = 2.0*pi * double(j-(Ny/2))/ Ly;
    
    fout_Amp << kx << " " << ky << " " << famp(Nxf-i,j) << endl;
    
        }
    
        for(int j =0; j< Ny/2; j++){
            
            kx = 2.0 * pi * double(i-Nxf+1) / Lx;
            ky = 2.0 * pi * double(j) / Ly;
            
            fout_Amp << kx << " " << ky << " " << famp(Nxf-i,j+(Ny/2)) << endl;
            
        }
        fout_Amp << endl;
        
    }
    
     for(int i =0; i < Nxf-1; i++){
    
         for(int j =0; j< Ny/2; j++){
             
             kx = 2.0*pi * double(i)/ Lx;
             ky = 2.0*pi * double(j-(Ny/2))/ Ly;
             
             fout_Amp << kx << " " << ky << " " << famp(i,j+(Ny/2)) << endl;
             
         }
         
         for(int j =0; j< Ny/2; j++){
             
             kx = 2.0*pi * double(i)/ Lx;
             ky = 2.0*pi * double(j)/ Ly;
             
             fout_Amp << kx << " " << ky << " " << famp(i,j) << endl;
             
         }
         fout_Amp << endl;
     }
          
         
    fout_Amp.close();

   return;
 
    }








