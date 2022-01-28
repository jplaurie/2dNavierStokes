
//this includes all the preconditoning plans for the funtions


#include <armadillo>
#include <complex>
#include <fftw3.h>

using namespace std;
using namespace arma;

void linear(cx_mat &);

cx_mat nonlinear(cx_mat , double, cx_mat ,mat &,cx_mat & ,fftw_plan ,fftw_plan);

void read(mat & , cx_mat &,double&, int&, fftw_plan);
void conservation_laws(double, int,cx_mat, double&);

void flux(double, cx_mat, cx_mat,rowvec &,rowvec &,int,int );

void spectrum( cx_mat, rowvec & ,rowvec &, int, int );

void forcing(cx_mat,mat);

void forcing_annulus(mat &, double, double);
void forcing_exponential(mat &, double, double);
void forcing_grid(mat &, double, double);
void output_w( cx_mat &, mat &, int,fftw_plan &);
void symmetry(cx_mat &);

void record_parameters();
void setup_ETDRK(cx_mat ,cx_mat &,cx_mat &,cx_mat &,cx_mat &, cx_mat &, cx_mat &,cx_mat &, cx_mat &, cx_mat &, cx_mat &, double);
