
//standalone file that creates an initial condition


#include <iostream>
#include <cmath>
#include <blitz/array.h>
#include <complex>
#include <fstream>


using namespace std;

const int Nx = 128;
const int Ny = 128;
const double pi = 3.14159265358979323846;
const double Lx = 2.0*pi;
const double dx= Lx/Nx;
const double Ly = 2.0*pi;
const double dy= Ly/Ny;

int main(){
	
    blitz::Array<double, 2> w(Nx,Ny);
    w = 0.0;


   
    ofstream fout("./w.initial");				//open up file
	fout << scientific;
	fout.precision(12);
        for(int j=0; j < Ny; j++){
             for(int i=0; i < Nx; i++){
      
            fout << w(i,j) << " ";
        }
        fout << endl;
    }
    fout.close();
 


    return 0;

}
