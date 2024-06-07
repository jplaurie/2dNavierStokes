
//=======SET RESOLUTION AND DOMAIN PARAMETERS====================

const int Nx = 1024;                     //number of spatial grid points

const int Ny = 1024;                      //number of spatial grid points

const double pi = 3.14159265358979323846;       //pi
const int Mx = 3*Nx/2;                      //defined the size of the anti-aliasing array M=2N
const int My = 3*Ny/2;                      //defined the size of the anti-aliasing array M=2N
const int Nxf = (Nx/2)+1;
const int Mxf = (Mx/2)+1;

const double asp_rat = 1.0;
const double Lx = 2.0*pi*asp_rat;		//2.0*pi*sqrt(asp_rat);                   //length of the box
const double dx = Lx/ double(Nx);                      //grid size
const double Ly = 2.0*pi;			//2.0*pi/sqrt(asp_rat);                   //length of the box
const double dy = Ly/ double(Ny);          

//============SET TIMESTEPPING PARAMETERS==============================

const bool FLAG_TIMESTEP_ETDRK = true; //if true routine is ETDRK4 else RK2
const int FLAG_ETDRK_ORDER = 4;
const double dt = 1.e-4;                //time step
const long int nsteps = 9999999999;                    //number of total time steps
const int outstep = 100;                    //outputs data at these time steps

//========SET EQUATION PARAMETERS=================================   

const bool FLAG_QG = false;   // turn this on to include beta effect
const double beta = 36.0;

const double nu = 1.e-41; // 1.e-36 for 512 1.e-28 for 128
const double nupower = 8.0;  //==  \nu (-\Delta)^nupower
const double alpha = 1.e-4;//3.e-5;
const double alphapower = 0.0;  // == \alpha (-\Delta)^alphapower

//=========SET FORCING PARAMETERS=====================================

const bool FLAG_FORCING_ON = true; //Flag to turn on forcing
const bool FLAG_FORCE_AMP_RESCALE = false; // Flag for having forcing amplitude rescaled to give unit energy density
const bool FLAG_FORCE_EXP = false;  // turn to true to have exponential forcing spectrum **default is annulus***
const bool FLAG_FORCE_STATIC = false; 
const double kf = 256.0;	//k_f=71 for static		// forcing scale
const double dk = 1.0;				//width of forcing band
const double Amp = 0.2;				// amplitude forcing
const double force_power = 4.0; // for exp force only f_k = (k/k_f)^force_power * exp( -(k / k_f)^force_power )

//===========OTHER PARAMETERS================

const int NBIN = fmax(Nx,Ny);
const bool FLAG_TIME_SERIES_MODES = false;
const int num_threads = 1;
