/******************************************************************** 
** This program solves the extended Hasegawa-Wakatani equations in 2D
** using finite differences with Arakawa scheme and 3rd order linear 
** multistep method with doubly periodic boundary conditions.
**
** Original: Joe Dewhurst 19/02/2008
**
** Modifications:
** 1. Use malloc not double 2d array declarations as these use stack
**    and array longer than 250 x 250 elements are not possible.
**
** Parallelisation(MPI) & netCDF output: Sanket Gadgil 12/08/2019
*********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <netcdf.h>
#include <mpi.h>

#define DEFAULT_SIZE 40.0
#define DEFAULT_GRID_SIZE 256
#define DEFAULT_WAVENUMBER 0.3
#define DEFAULT_V0 1.0
#define MAX(a,b) ((a)>(b) ? (a) : (b))

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

char make_2dArray(int grid_size, double **array, double *data);

int* netcdf_creator(char* file_name, int ndims, int grid_size, int nt);

int netcdf_meta(int ncid, int ndims, int* dimids, char* var);

void netcdf_endmeta(int ncid);

void netcdf_write1d(int ncid, int varid, double var, int k);

void netcdf_write2d(int ncid, int varid, double **var, int k, int grid_size);

void netcdf_close(int ncid);

double timer();

void coordinate_map(int **startx, int **starty, int **endx, int **endy, int **lengthx, int **lengthy, int p, int nl, int my_rank);

void sendrecv_alloc(double ***sendbuf, double ***recvbuf, double **sendbuf_data, double **recvbuf_data, int nl);

void distribute_to_root_rank(double ***data, int my_rank, double ***sendbuf, double ***recvbuf, int nl, int p, int **startx, int **starty, int **endx, int **endy);

int main(int argc, char *argv[])
{
	int my_rank, np, srce, dest;

	/*MPI Initialization majority of the code runs concurrently on all ranks*/
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	if (my_rank == 0) {printf("No. of processors: %d\n", np);}

	double start_t, t0, end_t, total_t, mark_0, mark_1, mark_2, mark_3;

	/*** constants ***/
	const double PI    = 3.14159265358979323846;
	const double c6d11 = 6.0/11.0;
	const double c1d12 = 1.0/12.0;
	const double c1d3  =  1.0/3.0;
	const double c1d2  =  1.0/2.0;
	const double c1d4  =  1.0/4.0;
	const double c1d8  =  1.0/8.0;
	const double c3d2  =  3.0/2.0;
	double c1dnl, c1dnl2;

	double A; A=0.5;		/*adiabaticity parameter*/
	double K; K=1.0;		/*density gradient driving parameter*/
	double D; D=0.0001;		/*density equation viscocity*/
	double M; M=0.0001;		/*vorticity equation viscocity*/
	double PN; PN=1.0;		/*density equation poisson bracket switch*/
	double PV; PV=1.0;		/*vorticity equation poisson bracket switch*/
	double ky;                  /*poloidal wavenumber of initial pertubation*/
	double ky_index;
	double V; double V0;

	double ZN; ZN=1.0;		/*zonal component damping parameter in density*/
	double ZP; ZP=1.0;		/*zonal component damping parameter in potential*/
	double EQ_T; EQ_T=0.250;     /* Threshold for the energy equipartition between ZF and turbulence */

	double C; C=0.0;//5;		/*curvature parameter for interchange drive; C=0 => HW case*/

	int nt; nt=3000; 		/*number of iterations*/
	int ti; ti=100;		/*output every ti iterations*/
	int out_t_st, out_t_end;    /* Output Start and end times for the simulation box */
	int printi; printi = 49500;	/*start output at printi'th iteration*/
	int nl;      	 	/*number of grid points*/

	double dt; dt=0.01;	        /*time step*/
	double cutoff; cutoff = /*0.0/dt;*/1200000.0/dt;
	double L;			/*domain size*/
	double dl, dl2;     	/*grid spacing and its square*/
	double dlinv, dl2inv;	/*inverse of grid spacing and its square*/
	double tr1,tr2,tr3,tr4,tr5;	/*temporary variables for therm in the equations*/
	char alloc_success;


	/* Arrays */
	double **n,**n1,**n2,**n3;	/* Current Density */
	double **fn1, **fn2, **fn3;	/* Previous Density */
	double **v,**v1,**v2,**v3;	/* Current Vorticity */
	double **fv1, **fv2, **fv3;	/* Previous Vorticity */
	double **phi;		/* Potential */
	double **phi_k;             /* Potential in k-space */
	double phi_k_s; phi_k_s = 0;             /* Potential in k-space at k=ky */
	double **nflux;		/*radial density flux*/

	/* 1D data equivalents */
	double *n_data, *n1_data, *n2_data, *n3_data;
	double *fn1_data, *fn2_data, *fn3_data;
	double *v_data, *v1_data, *v2_data, *v3_data;
	double *fv1_data, *fv2_data, *fv3_data;
	double *phi_data;
	double *phi_k_data;
	double *nflux_data;


	double *nfsa, *nfluxfsa, *phifsa, *vfsa;	/*flux surface averaged quantities*/


	/* Quantities related to energy conservation
	d(En+Egp)/dt = totalnflux + totalaflux + totalD + totalCnflux + totalCN and
	dW/dt = totalnflux + totalDW + totalCnflux
	*/
	double En=0.0, Egp=0.0;		/*n^2 energy, (grad phi)^2 energy*/
	double EgT=0.0, EgZ=0.0;		/*(gradphi)^2 energy for turb and zonal flows*/
	double W=0.0;			/*generalised enstrophy*/

	double Tnflux;		/*radial density flux integrated over all area*/
	double Taflux;		/*flux due to resistive dissipation through the parallel current*/

	double TD, TDW;		/*dissipation*/

	double TCnflux;		/*radial density flux due to interchange forcing*/
	double TCN;		        /*curvature terms*/

	/* Position on the grid */
	double *x, *y;
	int *mm, *nn;
	int i,j,k,l;

	/* Probes */
	double *nprobes, *nfsaprobes;
	double *nfluxprobes, *nfluxfsaprobes;
	double *phiprobes, *phifsaprobes;
	double *vortprobes, *vortfsaprobes;

	int px[9];
	int py; py = 128;
	px[0]= 0;
	for (i = 1; i<9; i++)
	{
		px[i] = (32*i) -1;
	}


	/*** FFTW ***/
	fftw_complex *in;
	fftw_plan p;
	fftw_plan q;

	char filename[128],commandname[128],varname[128];

	/*************************************************************
	**  C O D E   S T A R T S  H E R E
	*************************************************************/

	/* Validate command line arguments */
if (my_rank == 0)
{
	if(argc<4)
	{

		printf("\n*************************************************************\n");
		printf("** Usage: %s system_size no_of_grid_points [OPTIONS] **\n",argv[0]);
		printf("** Some input parameters missing...Assigning default values **\n");
		printf("\n*************************************************************\n");

		L=DEFAULT_SIZE;
		nl=DEFAULT_GRID_SIZE;
		ky=DEFAULT_WAVENUMBER;
		V0 = DEFAULT_V0;

	}
	else if(argc==4){
		V0 = DEFAULT_V0;
		sscanf(argv[1], "%lf", &L);
		sscanf(argv[2], "%d",  &nl);
		sscanf(argv[3], "%lf", &ky);

	}
	else if(argc>4){

		/* Copy arguments to variables */
		sscanf(argv[1], "%lf", &L);
		sscanf(argv[2], "%d",  &nl);
		sscanf(argv[3], "%lf", &ky);
		sscanf(argv[4], "%lf", &V0);
	}
}

	/*Broadcast initial input values to all ranks*/
	MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nl, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ky, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&V0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	dl=L/(double)nl;

	/*ky *= PI;*/
	ky *= (2.0*PI)/L;

	if(fmod(ky/dl, 1)<0.5){
		ky_index = (int)(ky/dl);
	}
	else{
		ky_index = (int)(ky/dl) + 1;
	}



	out_t_st=0;out_t_end = nt;

	/*Coordinate Map setup*/
	int *startx; int *starty; int *endx; int *endy; int *lengthx; int *lengthy;
	coordinate_map(&startx, &starty, &endx, &endy, &lengthx, &lengthy, np, nl, my_rank);

	/*Send and Recv Buffer allocation*/
	double *sendbuf_data; double *recvbuf_data;
	double **sendbuf; double **recvbuf;
	sendrecv_alloc(&sendbuf, &recvbuf, &sendbuf_data, &recvbuf_data, nl);

	/*Send and receive buffers for Enerfy calculation*/
	double E_sendbuf; double E_recvbuf;

	/*** Main variables allocated ***/
if (my_rank == 0)
{
	printf("\n*****************************************************************************\n");
	printf("** Allocating 2D arrays: Domain=%.1lf  Grid=%dx%d dl=%lf dt=%lf **\n",L,nl,nl,dl,dt);
	printf("** Physical Parameters: Parallel transport: A=%.2lf      Driving: K=%lf **\n",A,K);
	printf("*****************************************************************************\n");
}

	x = (double*)malloc(sizeof(double)*(nl+2));
	y = (double*)malloc(sizeof(double)*(nl+2));

	mm = (int*)malloc(sizeof(int)*(nl+1));
	nn = (int*)malloc(sizeof(int)*(nl+1));

	nfsa = (double*)malloc(sizeof(double)*(nl+2));
	nfluxfsa = (double*)malloc(sizeof(double)*(nl+2));
	phifsa = (double*)malloc(sizeof(double)*(nl+2));
	vfsa = (double*)malloc(sizeof(double)*(nl+2));

	/* 1D data array allocation */

	n_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	n1_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	n2_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	n3_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));

	fn1_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	fn2_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	fn3_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));

	v_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	v1_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	v2_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	v3_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));

	fv1_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	fv2_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	fv3_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));

	phi_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	phi_k_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));
	nflux_data = (double*)malloc((nl+2)*(nl+2)*sizeof(double));

	/* Probes allocation */
	nprobes = (double*)malloc(9*sizeof(double));
	nfsaprobes = (double*)malloc(9*sizeof(double));
	nfluxprobes = (double*)malloc(9*sizeof(double));
	nfluxfsaprobes = (double*)malloc(9*sizeof(double));
	phiprobes = (double*)malloc(9*sizeof(double));
	phifsaprobes = (double*)malloc(9*sizeof(double));
	vortprobes = (double*)malloc(9*sizeof(double));
	vortfsaprobes = (double*)malloc(9*sizeof(double));

	/* 2D array allocation */
	n=(double**)malloc((nl+2)*sizeof(double *));
	if(n==NULL) {printf("Allocation of n failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, n, n_data);
	if(!alloc_success) {printf("Allocation of n failed\n"); exit(EXIT_FAILURE);}

	n1=(double**)malloc((nl+2)*sizeof(double *));
	if(n1==NULL) {printf("Allocation of n1 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, n1, n1_data);
	if(!alloc_success) {printf("Allocation of n1 failed\n"); exit(EXIT_FAILURE);}

	n2=(double**)malloc((nl+2)*sizeof(double *));
	if(n2==NULL) {printf("Allocation of n2 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, n2, n2_data);
	if(!alloc_success) {printf("Allocation of n2 failed\n"); exit(EXIT_FAILURE);}

	n3=(double**)malloc((nl+2)*sizeof(double *));
	if(n3==NULL) {printf("Allocation of n3 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, n3, n3_data);
	if(!alloc_success) {printf("Allocation of n3 failed\n"); exit(EXIT_FAILURE);}

	fn1=(double**)malloc((nl+2)*sizeof(double *));
	if(fn1==NULL) {printf("Allocation of fn1 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, fn1, fn1_data);
	if(!alloc_success) {printf("Allocation of fn1 failed\n"); exit(EXIT_FAILURE);}

	fn2=(double**)malloc((nl+2)*sizeof(double *));
	if(fn2==NULL) {printf("Allocation of fn2 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, fn2, fn2_data);
	if(!alloc_success) {printf("Allocation of fn2 failed\n"); exit(EXIT_FAILURE);}

	fn3=(double**)malloc((nl+2)*sizeof(double *));
	if(fn3==NULL) {printf("Allocation of fn3 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, fn3, fn3_data);
	if(!alloc_success) {printf("Allocation of fn3 failed\n"); exit(EXIT_FAILURE);}

	v=(double**)malloc((nl+2)*sizeof(double *));
	if(v==NULL) {printf("Allocation of v failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, v, v_data);
	if(!alloc_success) {printf("Allocation of v failed\n"); exit(EXIT_FAILURE);}

	v1=(double**)malloc((nl+2)*sizeof(double *));
	if(v1==NULL) {printf("Allocation of v1 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, v1, v1_data);
	if(!alloc_success) {printf("Allocation of v1 failed\n"); exit(EXIT_FAILURE);}

	v2=(double**)malloc((nl+2)*sizeof(double *));
	if(v2==NULL) {printf("Allocation of v2 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, v2, v2_data);
	if(!alloc_success) {printf("Allocation of v2 failed\n"); exit(EXIT_FAILURE);}

	v3=(double**)malloc((nl+2)*sizeof(double *));
	if(v3==NULL) {printf("Allocation of v3 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, v3, v3_data);
	if(!alloc_success) {printf("Allocation of v3 failed\n"); exit(EXIT_FAILURE);}

	fv1=(double**)malloc((nl+2)*sizeof(double *));
	if(fv1==NULL) {printf("Allocation of fv1 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, fv1, fv1_data);
	if(!alloc_success) {printf("Allocation of fv1 failed\n"); exit(EXIT_FAILURE);}

	fv2=(double**)malloc((nl+2)*sizeof(double *));
	if(fv2==NULL) {printf("Allocation of fv2 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, fv2, fv2_data);
	if(!alloc_success) {printf("Allocation of fv2 failed\n"); exit(EXIT_FAILURE);}

	fv3=(double**)malloc((nl+2)*sizeof(double *));
	if(fv3==NULL) {printf("Allocation of fv3 failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, fv3, fv3_data);
	if(!alloc_success) {printf("Allocation of fv3 failed\n"); exit(EXIT_FAILURE);}

	phi=(double**)malloc((nl+2)*sizeof(double *));
	if(phi==NULL) {printf("Allocation of phi failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, phi, phi_data);
	if(!alloc_success) {printf("Allocation of phi failed\n"); exit(EXIT_FAILURE);}

	phi_k = (double**)malloc((nl+2)*sizeof(double *));
	if(phi_k==NULL) {printf("Allocation of phi_k failed\n"); exit(EXIT_FAILURE);}
	else alloc_success=make_2dArray(nl+2, phi_k, phi_k_data);
	if(!alloc_success) {printf("Allocation of phi_k failed\n"); exit(EXIT_FAILURE);}

	nflux=(double**)malloc((nl+2)*sizeof(double *));
	if(nflux==NULL) {printf("Allocation of nflux failed\n"); exit(EXIT_FAILURE);}
	else  alloc_success=make_2dArray(nl+2, nflux, nflux_data);
	if(!alloc_success) {printf("Allocation of nflux failed\n"); exit(EXIT_FAILURE);}

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nl * nl);
	p = fftw_plan_dft_2d(nl, nl, in, in, FFTW_FORWARD, FFTW_MEASURE);
	q = fftw_plan_dft_2d(nl, nl, in, in, FFTW_BACKWARD, FFTW_MEASURE);

	c1dnl = 1.0/(double)nl;
	c1dnl2 = 1.0/((double)nl*(double)nl);

if (my_rank == 0)
{
	printf("\n******************************************\n");
	printf("** All 2D arrays allocated successfully **\n");
	printf("******************************************\n");
}

	/* Assign mode numbers, 0 to nl */
	l = 0;
	for(i = 1; i < nl/2+1; i++)
	{
		mm[i] = i;
		nn[i] = i;
 		l += 1;
	}

	for(i = nl/2+1; i < nl+1; i++)
	{
		mm[i] = nl+1-i;
		nn[i] = nl+1-i;
		l += 1;
	}

	/* Set up grid */
	x[0] = -0.5*dl*(nl + 1);
	y[0] = -0.5*dl*(nl + 1);
	for(i = 0; i < nl+1; i++) x[i+1] = x[i] + dl;
	for(j = 0; j < nl+1; j++) y[j+1] = y[j] + dl;

	/* Initial conditions */

	for(i = 0; i < nl+2; i++)
	{
		for(j = 0; j < nl+2; j++)
		{
			double kx; kx = 0.0;//1*ky;/*minimum of 0.15625(wavelength of 4.02)*/
			double lambda; lambda = 0.4*PI;/*1.256637(wavelength of 5.0)*/
			double plambda; plambda = 10.0*lambda;/*1.570796(wavelength of 4.0)*/
			double VT; VT = 0.0;
			double VDW; VDW = 0.01*(-V0/lambda);
			n[i][j] =  VDW*cos(kx*x[i] + ky*y[j]);
			n1[i][j] = 0.0;
			n2[i][j] = 0.0;
			n3[i][j] = 0.0;
			fn1[i][j] = 0.0;
			fn2[i][j] = 0.0;
			fn3[i][j] = 0.0;
			v[i][j] = (lambda*V0)*cos(lambda*x[i]) - (kx*kx + ky*ky)*VDW*cos(kx*x[i]+ky*y[j]);
			v1[i][j] = 0.0;
			v2[i][j] = 0.0;
			v3[i][j] = 0.0;
			fv1[i][j] = 0.0;
			fv2[i][j] = 0.0;
			fv3[i][j] = 0.0;
			phi[i][j] = -(V0/lambda)*cos(lambda*x[i]) + VDW*cos(kx*x[i] + ky*y[j]);

			phi_k[i][j] = 0.0;

			nflux[i][j] = 0.0;
		}
	}

	/*** Energy and flux and enstrophy***/
	En = 0.0; Egp = 0.0; EgT = 0.0; EgZ = 0.0; W = 0.0; TD = 0.0; TDW = 0.0;
	TCnflux = 0.0; Tnflux = 0.0; Taflux = 0.0; TCN = 0.0;

	/* Helpful quantities */
	dl2=dl*dl; dlinv=1.0/dl; dl2inv=1.0/dl2;

	for(i = 0; i < nl+2; i++)
	{
		nfsa[i] = 0.0;
		nfluxfsa[i] = 0.0;
		phifsa[i] = 0.0;
		vfsa[i] = 0.0;
	}


	/*Initial Energy calculations*/
	for(i = 1; i < nl+1; i++)
	{
		for(j = 1; j < nl+1; j++)
		{
			En += c1d2*dl2*(n[i][j]*n[i][j]);

			Egp += c1d8*(phi[i+1][j]-phi[i-1][j])*(phi[i+1][j]-phi[i-1][j])
			+ c1d8*(phi[i][j+1]-phi[i][j-1])*(phi[i][j+1]-phi[i][j-1]);

			W += c1d2*dl2*(n[i][j]-v[i][j])*(n[i][j]-v[i][j]);

			nflux[i][j] = -c1d2*K*n[i][j]*dlinv*(phi[i][j+1]-phi[i][j-1]);

			Tnflux += dl2*nflux[i][j];

			TD += D*n[i][j]*(n[i+1][j] + n[i-1][j] - 4*n[i][j] + n[i][j+1] + n[i][j-1])\
					- M*phi[i][j]*(v[i+1][j] + v[i-1][j] - 4*v[i][j] + v[i][j+1] + v[i][j-1]);

			TDW += (n[i][j]-v[i][j])*(D*(n[i+1][j] + n[i-1][j] - 4*n[i][j] + n[i][j+1] + n[i][j-1])\
					- M*(v[i+1][j] + v[i-1][j]- 4*v[i][j] + v[i][j+1] + v[i][j-1]));

			TCnflux += c1d2*dl*C*n[i][j]*(phi[i][j+1]-phi[i][j-1]);

			TCN += c1d2*dl*C*(n[i][j]+phi[i][j])*(n[i][j+1]-n[i][j-1]);
			nfsa[i] += n[i][j]/nl;
			nfluxfsa[i] += nflux[i][j]/nl;
			phifsa[i] += phi[i][j]/nl;
			vfsa[i] += v[i][j]/nl;
		}
	}

	for(i = 1; i < nl+1; i++)
	{
		for(j = 1; j < nl+1; j++)
		{
			EgT += c1d8*((phi[i+1][j]-phifsa[i+1])-(phi[i-1][j]-phifsa[i-1]))*
	          			((phi[i+1][j]-phifsa[i+1])-(phi[i-1][j]-phifsa[i-1])) +
	     			c1d8*((phi[i][j+1]-phifsa[i])-(phi[i][j-1]-phifsa[i]))*
	          			((phi[i][j+1]-phifsa[i])-(phi[i][j-1]-phifsa[i]));

			EgZ += c1d8*(phifsa[i+1]-phifsa[i-1])*(phifsa[i+1]-phifsa[i-1]);

			Taflux += -dl2*A*((phi[i][j]-ZP*phifsa[i])-(n1[i][j]-ZN*nfsa[i]))*
	               			((phi[i][j]-ZP*phifsa[i])-(n1[i][j]-ZN*nfsa[i]));
		}
	}

	for (i = 0; i < 9; i++)
	{
		nprobes[i] = n[px[i]][py];
		nfsaprobes[i] = nfsa[px[i]];

		nfluxprobes[i] = nflux[px[i]][py];
		nfluxfsaprobes[i] = nfluxfsa[px[i]];

		phiprobes[i] = phi[px[i]][py];
		phifsaprobes[i] = phifsa[px[i]];

		vortprobes[i] = v[px[i]][py];
		vortfsaprobes[i] = vfsa[px[i]];
	}

	/*** netCDF variables ***/

	int ndims0, ndims1, ndims2, ncid0, ncid1, ncid2;
	int *markers0, *markers1, *markers2, *dimids0, *dimids1, *dimids2;

	int nprobes_id[9];	int nfsaprobes_id[9];
	int nfluxprobes_id[9]; int nfluxfsaprobes_id[9];
	int phiprobes_id[9]; int phifsaprobes_id[9];
	int vortprobes_id[9]; int vortfsaprobes_id[9];

if (my_rank == 0)
{
	/*** netCDF Probes output setup ***/

	sprintf(filename, "variable_probes_A%.2lf.nc", A);
	ndims0= 1;
	markers0 = (int *)malloc((1+ndims0)*sizeof(int));
	markers0 = netcdf_creator(filename, ndims0, (nl+2), (nt+1));
	ncid0 = markers0[0];
	dimids0 = (int *)malloc((1+ndims0)*sizeof(int));
	dimids0[0] = markers0[1];
	free(markers0);

	for (i = 0; i < 9; i++)
	{
		sprintf(varname, "nprobe_%d", i);
		nprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);

		sprintf(varname, "nfsaprobe_%d", i);
		nfsaprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);

		sprintf(varname, "nfluxprobe_%d", i);
		nfluxprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);

		sprintf(varname, "nfluxfsaprobe_%d", i);
		nfluxfsaprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);

		sprintf(varname, "phiprobe_%d", i);
		phiprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);

		sprintf(varname, "phifsaprobe_%d", i);
		phifsaprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);

		sprintf(varname, "vortprobe_%d", i);
		vortprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);

		sprintf(varname, "vortfsaprobe_%d", i);
		vortfsaprobes_id[i] = netcdf_meta(ncid0, ndims0, dimids0, varname);
	}

	netcdf_endmeta(ncid0);
	free(dimids0);

	for (i = 0; i < 9; i++)
	{
		netcdf_write1d(ncid0, nprobes_id[i], nprobes[i], 0);
		netcdf_write1d(ncid0, nfsaprobes_id[i], nfsaprobes[i], 0);

		netcdf_write1d(ncid0, nfluxprobes_id[i], nfluxprobes[i], 0);
		netcdf_write1d(ncid0, nfluxfsaprobes_id[i], nfluxfsaprobes[i], 0);

		netcdf_write1d(ncid0, phiprobes_id[i], phiprobes[i], 0);
		netcdf_write1d(ncid0, phifsaprobes_id[i], phifsaprobes[i], 0);

		netcdf_write1d(ncid0, vortprobes_id[i], vortprobes[i], 0);
		netcdf_write1d(ncid0, vortfsaprobes_id[i], vortfsaprobes[i], 0);
	}
}

int En_id; int Egp_id; int EgT_id; int EgZ_id; int W_id; int Tnflux_id; int Taflux_id; int TD_id;
int TDW_id; int TCnflux_id; int TCN_id; int phi_k_s_id;

if (my_rank == 0)
{
	/*** netCDF 1D output setup ***/
	sprintf(filename, "1d_variables.nc");
	ndims1= 1;
	markers1 = (int *)malloc((1+ndims1)*sizeof(int));
	markers1 = netcdf_creator(filename, ndims1, (nl+2), (nt+1));
	ncid1 = markers1[0];
	dimids1 = (int *)malloc((1+ndims1)*sizeof(int));
	dimids1[0] = markers1[1];
	free(markers1);

	sprintf(varname, "En");
	En_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "Egp");
	Egp_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "EgT");
	EgT_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "EgZ");
	EgZ_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "W");
	W_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "Tnflux");
	Tnflux_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "Taflux");
	Taflux_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "TD");
	TD_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "TDW");
	TDW_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "TCnflux");
	TCnflux_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "TCN");
	TCN_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	sprintf(varname, "phi_k_s");
	phi_k_s_id = netcdf_meta(ncid1, ndims1, dimids1, varname);

	netcdf_endmeta(ncid1);
	free(dimids1);

	netcdf_write1d(ncid1, En_id, En, 0);
	netcdf_write1d(ncid1, Egp_id, Egp, 0);
	netcdf_write1d(ncid1, EgT_id, EgT, 0);
	netcdf_write1d(ncid1, EgZ_id, EgZ, 0);
	netcdf_write1d(ncid1, W_id, W, 0);
	netcdf_write1d(ncid1, Tnflux_id, Tnflux, 0);
	netcdf_write1d(ncid1, Taflux_id, Taflux, 0);
	netcdf_write1d(ncid1, TD_id, TD, 0);
	netcdf_write1d(ncid1, TDW_id, TDW, 0);
	netcdf_write1d(ncid1, TCnflux_id, TCnflux, 0);
	netcdf_write1d(ncid1, TCN_id, TCN, 0);
	netcdf_write1d(ncid1, phi_k_s_id, phi_k_s, 0);
}

int n_id; int phi_id; int phi_k_id; int v_id;

if (my_rank == 0)
{
	/*** netCDF 2D output setup ***/
	sprintf(filename, "2d_variables.nc");
	ndims2 = 3;
	markers2 = (int *)malloc((1+ndims2)*sizeof(int));
	markers2 = netcdf_creator(filename, ndims2, (nl+2), (int)((double)nt/(double)ti));
	ncid2 = markers2[0];
	dimids2 = (int *)malloc((ndims2)*sizeof(int));
	dimids2[0] = markers2[1];	dimids2[1] = markers2[2];	dimids2[2] = markers2[3];
	free(markers2);


	sprintf(varname, "n");
	n_id = netcdf_meta(ncid2, ndims2, dimids2, varname);

	sprintf(varname, "phi");
	phi_id = netcdf_meta(ncid2, ndims2, dimids2, varname);

	sprintf(varname, "phi_k");
	phi_k_id = netcdf_meta(ncid2, ndims2, dimids2, varname);

	sprintf(varname, "v");
	v_id = netcdf_meta(ncid2, ndims2, dimids2, varname);

	netcdf_endmeta(ncid2);
	free(dimids2);


	netcdf_write2d(ncid2, phi_id, phi, 0, (nl+2));
	netcdf_write2d(ncid2, n_id, n, 0, (nl+2));
	netcdf_write2d(ncid2, phi_k_id, phi_k, 0, (nl+2));
	netcdf_write2d(ncid2, v_id, v, 0, (nl+2));
}

	/*Timing setup*/
	if (my_rank == 0) {start_t = timer(); t0 = start_t;}

	if (my_rank == 0) {mark_3=mark_2=mark_1=mark_0=start_t;}

if (my_rank == 0)
{
	printf("Timings and other diagnostics\n");
	printf("%-14s %-14s %-14s %-14s %-14s %-14s %-14s %-14s %-14s %-14s\n","Reassignment","Evolution","Fourier","Energies","I/O","Total Time","ZF indicate","Progress","No. Procs","ETA");
}

	/* Main loop start here */

	for(k = 1; k < nt+1; k++)
	{
		if(k%ti==0 && my_rank==0) start_t = timer();

		/*Broadcast n, v, fn1, fv1 to all ranks*/
		MPI_Bcast(&(n[0][0]), (nl+2)*(nl+2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(v[0][0]), (nl+2)*(nl+2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(fn1[0][0]), (nl+2)*(nl+2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(fv1[0][0]), (nl+2)*(nl+2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(nflux[0][0]), (nl+2)*(nl+2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

		/*Synchronize all ranks*/
		MPI_Barrier(MPI_COMM_WORLD);


		/*Dummy variables holding previous iteration values of
		  n, v, fn1 and fv1*/
		memcpy(&(n3[0][0]), &(n2[0][0]), (nl+2)*(nl+2)*sizeof(double));
		memcpy(&(n2[0][0]), &(n1[0][0]), (nl+2)*(nl+2)*sizeof(double));
		memcpy(&(n1[0][0]), &(n[0][0]), (nl+2)*(nl+2)*sizeof(double));

		memcpy(&(fn3[0][0]), &(fn2[0][0]), (nl+2)*(nl+2)*sizeof(double));
		memcpy(&(fn2[0][0]), &(fn1[0][0]), (nl+2)*(nl+2)*sizeof(double));

		memcpy(&(v3[0][0]), &(v2[0][0]), (nl+2)*(nl+2)*sizeof(double));
		memcpy(&(v2[0][0]), &(v1[0][0]), (nl+2)*(nl+2)*sizeof(double));
		memcpy(&(v1[0][0]), &(v[0][0]), (nl+2)*(nl+2)*sizeof(double));

		memcpy(&(fv3[0][0]), &(fv2[0][0]), (nl+2)*(nl+2)*sizeof(double));
		memcpy(&(fv2[0][0]), &(fv1[0][0]), (nl+2)*(nl+2)*sizeof(double));



		if(k%ti==0 && my_rank==0) mark_0 = timer();
		if(k>=cutoff){ PN = 0.0; PV = 0.0; }

		/*Main calculation loop*/
		/*The most expedient way to implement MPI without changing the core structure of the code
		  would be to simply have each rank work on a small portion of the full grid which they have a copy
		  of*/
		for(i = startx[my_rank]; i < endx[my_rank]; i++)
		{
			tr1=0; tr2=0;tr3=0;tr4=0;tr5=0;
			for(j = starty[my_rank]; j < endy[my_rank]; j++)
			{
				/* Evolve density */
				tr1 = -c1d2*dlinv*K*(phi[i][j+1]-phi[i][j-1]);

				tr2 = A*((phi[i][j]-ZP*phifsa[i])-(n1[i][j]-ZN*nfsa[i]));

				tr3 = D*dl2inv*(n1[i+1][j] + n1[i-1][j] - 4*n1[i][j] + n1[i][j+1] + n1[i][j-1]);

				tr4 = -(c1d12*PN*dl2inv)*( ((phi[i+1][j]-phi[i-1][j])*(n1[i][j+1]-n1[i][j-1]) \
                            					-(phi[i][j+1]-phi[i][j-1])*(n1[i+1][j]-n1[i-1][j])) \
                               					+(phi[i+1][j]*(n1[i+1][j+1]-n1[i+1][j-1]) \
												- phi[i-1][j]*(n1[i-1][j+1]-n1[i-1][j-1]) \
												- phi[i][j+1]*(n1[i+1][j+1]-n1[i-1][j+1]) \
												+ phi[i][j-1]*(n1[i+1][j-1]-n1[i-1][j-1])) \
												+(phi[i+1][j+1]*(n1[i][j+1]-n1[i+1][j]) \
												- phi[i-1][j-1]*(n1[i-1][j]-n1[i][j-1]) \
												- phi[i-1][j+1]*(n1[i][j+1]-n1[i-1][j]) \
												+ phi[i+1][j-1]*(n1[i+1][j]-n1[i][j-1])) );

				tr5 = c1d2*C*dlinv*(phi[i][j+1]-n1[i][j+1]-phi[i][j-1]+n1[i][j-1]);

 				fn1[i][j] = tr1+tr2+tr3+tr4+tr5;

 				n[i][j] = c6d11*(3*n1[i][j]-c3d2*n2[i][j]+c1d3*n3[i][j] \
		    				+dt*(3*fn1[i][j]-3*fn2[i][j]+fn3[i][j]));

 				/* Evolve vorticity */
 				tr1 = A*((phi[i][j]-ZP*phifsa[i])-(n1[i][j]-ZN*nfsa[i]));

				tr2 = M*dl2inv*(v1[i+1][j] + v1[i-1][j] - 4*v1[i][j] + v1[i][j+1] + v1[i][j-1]);

				tr3 = -(c1d12*PV*dl2inv)*( ((phi[i+1][j]-phi[i-1][j])*(v1[i][j+1]-v1[i][j-1]) \
                               					-(phi[i][j+1]-phi[i][j-1])*(v1[i+1][j]-v1[i-1][j]))\
                              					+(phi[i+1][j]*(v1[i+1][j+1]-v1[i+1][j-1])\
                               					- phi[i-1][j]*(v1[i-1][j+1]-v1[i-1][j-1])\
                               					- phi[i][j+1]*(v1[i+1][j+1]-v1[i-1][j+1])\
                               					+ phi[i][j-1]*(v1[i+1][j-1]-v1[i-1][j-1]))\
                               					+(phi[i+1][j+1]*(v1[i][j+1]-v1[i+1][j])\
                               					- phi[i-1][j-1]*(v1[i-1][j]-v1[i][j-1])\
                               					- phi[i-1][j+1]*(v1[i][j+1]-v1[i-1][j])\
                               					+ phi[i+1][j-1]*(v1[i+1][j]-v1[i][j-1])) );

				tr4 = c1d2*C*dlinv*(-n1[i][j+1]+n1[i][j-1]);

				fv1[i][j] = tr1+tr2+tr3+tr4;

				v[i][j] = c6d11*(3*v1[i][j]-c3d2*v2[i][j]+c1d3*v3[i][j] \
                             				+dt*(3*fv1[i][j]-3*fv2[i][j]+fv3[i][j]));

			}
		}

		/*Distribute n, v, fn1, fv1 to all ranks*/
		distribute_to_root_rank(&n, my_rank, &sendbuf, &recvbuf, nl, np, &startx, &starty, &endx, &endy);
		distribute_to_root_rank(&v, my_rank, &sendbuf, &recvbuf, nl, np, &startx, &starty, &endx, &endy);
		distribute_to_root_rank(&fn1, my_rank, &sendbuf, &recvbuf, nl, np, &startx, &starty, &endx, &endy);
		distribute_to_root_rank(&fv1, my_rank, &sendbuf, &recvbuf, nl, np, &startx, &starty, &endx, &endy);

		/*Synchronize all ranks*/
		MPI_Barrier(MPI_COMM_WORLD);

		/* Set boundary conditions (doubly periodic)*/
		for(j = 0; j < nl+2; j++)
		{
			n[0][j] = n[nl][j]; n[nl+1][j] = n[1][j];
			v[0][j] = v[nl][j]; v[nl+1][j] = v[1][j];
		}
		for(i = 0; i < nl+2; i++)
		{
			n[i][0] = n[i][nl]; n[i][nl+1] = n[i][1];
			v[i][0] = v[i][nl]; v[i][nl+1] = v[i][1];
		}

		/* Solve Laplace equation to calculate phi from v using FFT */
		/*input real*/
		for(j = 1; j < nl+1; j++)
		{
			for(i = 1; i < nl+1; i++)
			{
				l = (j-1)*nl+i-1;
				in[l][0] = v[i][j];
			}
		}

		/*input imaginary*/
		for(j = 1; j < nl+1; j++)
		{
			for(i = 1; i < nl+1; i++)
			{
				l = (j-1)*nl+i-1;
				in[l][1] = 0.0;
			}
		}

		/*FFT v*/
		fftw_execute(p);
		if(k%ti==0 && my_rank==0) mark_1=timer();
		/*solve for PHI in fourier space*/

		for(j = 1; j < nl+1; j++)
		{
			for(i = 1; i < nl+1; i++)
			{
				l = (j-1)*nl+i-1;
				in[l][0] *= dl2/(2*cos(2*PI*mm[i]*c1dnl)+2*cos(2*PI*nn[j]*c1dnl)-4);
				in[l][1] *= dl2/(2*cos(2*PI*mm[i]*c1dnl)+2*cos(2*PI*nn[j]*c1dnl)-4);
				phi_k[i][j] = in[l][0]*c1dnl2;
			}
		}

		/*set (m,n) = (0,0) component to zero*/
		in[0][0] = 0.0; in[0][1] = 0.0;
		phi_k_s = phi_k[1][(int)ky_index+1];

		/*inverse FFT*/
		fftw_execute(q);

		/*output real*/
		for(j = 1; j < nl+1; j++)
		{
			for(i = 1; i < nl+1; i++)
			{
				l = (j-1)*nl+i-1;
				phi[i][j] = in[l][0]*c1dnl2;
			}
		}

		/*boundary conditions (doubly periodic)*/
		for(j = 0; j < nl+2; j++)
		{
			phi[0][j] = phi[nl][j];
			phi[nl+1][j] = phi[1][j];

			phi_k[0][j] = phi_k[nl][j];
			phi_k[nl+1][j] = phi_k[1][j];
		}

		for(i = 0; i < nl+2; i++)
		{
			phi[i][0] = phi[i][nl];
			phi[i][nl+1] = phi[i][1];

			phi_k[i][0] = phi_k[i][nl];
			phi_k[i][nl+1] = phi_k[i][1];
		}

		if(k%ti==0 && my_rank==0) mark_2=timer();

		/* Energy, flux and enstrohpy */
		En = 0.0; Egp = 0.0; W = 0.0; EgZ = 0.0; EgT = 0.0;
		Tnflux = 0.0; Taflux = 0.0; TD = 0.0; TDW = 0.0; TCnflux = 0.0; TCN = 0.0;

		nfsa[0] = 0.0; nfsa[nl+1] = 0.0;
		nfluxfsa[0] = 0.0; nfluxfsa[nl+1] = 0.0;
		phifsa[0] = 0.0; phifsa[nl+1] = 0.0;
		vfsa[0] = 0.0; vfsa[nl+1] = 0.0;

		for(i = 1; i < nl+1; i++)
		{
			nfsa[i] = 0.0;
			nfluxfsa[i] = 0.0;
			phifsa[i] = 0.0;
			vfsa[i] = 0.0;

			for(j = 1; j < nl+1; j++)
			{
				nfsa[i] += n[i][j]*c1dnl;
				nfluxfsa[i] += nflux[i][j]*c1dnl;
				phifsa[i] += phi[i][j]*c1dnl;
				vfsa[i] += v[i][j]*c1dnl;
			}
		}

		/*MPI parallelization of Energy calculations performed in similar
		  fashion as above.*/
		for(i = startx[my_rank]; i < endx[my_rank]; i++)
		{
			for(j = starty[my_rank]; j < endy[my_rank]; j++)
			{
				En += c1d2*dl2*(n[i][j]*n[i][j]);

				Egp += c1d8*(phi[i+1][j]-phi[i-1][j])*(phi[i+1][j]-phi[i-1][j])
				+ c1d8*(phi[i][j+1]-phi[i][j-1])*(phi[i][j+1]-phi[i][j-1]);

				W += c1d2*dl2*(n[i][j]-v[i][j])*(n[i][j]-v[i][j]);

				nflux[i][j] = -c1d2*K*n[i][j]*dlinv*(phi[i][j+1]-phi[i][j-1]);

				Tnflux += dl2*nflux[i][j];

				TD += D*n[i][j]*(n[i+1][j] + n[i-1][j] - 4*n[i][j] + n[i][j+1] + n[i][j-1]) \
    				- M*phi[i][j]*(v[i+1][j] + v[i-1][j] - 4*v[i][j] + v[i][j+1] + v[i][j-1]);

				TDW += (n[i][j]-v[i][j])*(D*(n[i+1][j] + n[i-1][j] - 4*n[i][j] + n[i][j+1] + n[i][j-1])\
		        				- M*(v[i+1][j] + v[i-1][j]- 4*v[i][j] + v[i][j+1] + v[i][j-1]));

				TCnflux += c1d2*dl*C*n[i][j]*(phi[i][j+1]-phi[i][j-1]);

				TCN += c1d2*dl*C*(n[i][j]+phi[i][j])*(n[i][j+1]-n[i][j-1]);

				EgT += c1d8*((phi[i+1][j]-phifsa[i+1])-(phi[i-1][j]-phifsa[i-1]))*
	           				((phi[i+1][j]-phifsa[i+1])-(phi[i-1][j]-phifsa[i-1])) +
	      				c1d8*((phi[i][j+1]-phifsa[i])-(phi[i][j-1]-phifsa[i]))*
	           				((phi[i][j+1]-phifsa[i])-(phi[i][j-1]-phifsa[i]));

				EgZ += c1d8*(phifsa[i+1]-phifsa[i-1])*(phifsa[i+1]-phifsa[i-1]);

				Taflux += -dl2*A*((phi[i][j]-ZP*phifsa[i])-(n1[i][j]-ZN*nfsa[i]))*
	                				((phi[i][j]-ZP*phifsa[i])-(n1[i][j]-ZN*nfsa[i]));
			}
		}

		distribute_to_root_rank(&nflux, my_rank, &sendbuf, &recvbuf, nl, np, &startx, &starty, &endx, &endy);

		/*MPI_Allreduce used to sum and broadcast values to all ranks*/
		E_sendbuf = En;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		En = E_recvbuf;

		E_sendbuf = W;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		W = E_recvbuf;

		E_sendbuf = Tnflux;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		Tnflux = E_recvbuf;

		E_sendbuf = TD;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		TD = E_recvbuf;

		E_sendbuf = TDW;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		TDW = E_recvbuf;

		E_sendbuf = TCnflux;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		TCnflux = E_recvbuf;

		E_sendbuf = TCN;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		TCN = E_recvbuf;

		E_sendbuf = EgT;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		EgT = E_recvbuf;

		E_sendbuf = EgZ;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		EgZ = E_recvbuf;

		E_sendbuf = Taflux;
		MPI_Allreduce(&E_sendbuf, &E_recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		Taflux = E_recvbuf;


		/*Synchronize all ranks*/
		MPI_Barrier(MPI_COMM_WORLD);

		for (i = 0; i < 9; i++)
		{
			nprobes[i] = n[px[i]][py];
			nfsaprobes[i] = nfsa[px[i]];

			nfluxprobes[i] = nflux[px[i]][py];
			nfluxfsaprobes[i] = nfluxfsa[px[i]];

			phiprobes[i] = phi[px[i]][py];
			phifsaprobes[i] = phifsa[px[i]];

			vortprobes[i] = v[px[i]][py];
			vortfsaprobes[i] = vfsa[px[i]];
		}


		if(k%ti==0 && my_rank==0) mark_3 = timer();

if (my_rank == 0)
{
		/* Output 2D quantities */
		if((k >= out_t_st)&&(k<out_t_end)&&(k%ti==0))
		{
			netcdf_write2d(ncid2, n_id, n, (int)((double)k/(double)ti), (nl+2));
			netcdf_write2d(ncid2, phi_id, phi, (int)((double)k/(double)ti), (nl+2));
			netcdf_write2d(ncid2, phi_k_id, phi_k, (int)((double)k/(double)ti), (nl+2));
			netcdf_write2d(ncid2, v_id, v, (int)((double)k/(double)ti), (nl+2));
		}

		/* Output probes */
		for (i = 0; i < 9; i++)
		{
			netcdf_write1d(ncid0, nprobes_id[i], nprobes[i], k);
			netcdf_write1d(ncid0, nfsaprobes_id[i], nfsaprobes[i], k);

			netcdf_write1d(ncid0, nfluxprobes_id[i], nfluxprobes[i], k);
			netcdf_write1d(ncid0, nfluxfsaprobes_id[i], nfluxfsaprobes[i], k);

			netcdf_write1d(ncid0, phiprobes_id[i], phiprobes[i], k);
			netcdf_write1d(ncid0, phifsaprobes_id[i], phifsaprobes[i], k);

			netcdf_write1d(ncid0, vortprobes_id[i], vortprobes[i], k);
			netcdf_write1d(ncid0, vortfsaprobes_id[i], vortfsaprobes[i], k);
		}

		/* Output 1D quantities */
		netcdf_write1d(ncid1, En_id, En, k);
		netcdf_write1d(ncid1, Egp_id, Egp, k);
		netcdf_write1d(ncid1, EgT_id, EgT, k);
		netcdf_write1d(ncid1, EgZ_id, EgZ, k);
		netcdf_write1d(ncid1, W_id, W, k);
		netcdf_write1d(ncid1, Tnflux_id, Tnflux, k);
		netcdf_write1d(ncid1, Taflux_id, Taflux, k);
		netcdf_write1d(ncid1, TD_id, TD, k);
		netcdf_write1d(ncid1, TDW_id, TDW, k);
		netcdf_write1d(ncid1, TCnflux_id, TCnflux, k);
		netcdf_write1d(ncid1, TCN_id, TCN, k);
		netcdf_write1d(ncid1, phi_k_s_id, phi_k_s, k);
}
		/*Timings and other diagnostics*/

		if(k%ti==0 && my_rank==0) { end_t = timer();
			printf("\r%-14lf %-14lf %-14lf %-14lf %-14lf %-14lf %-14lf %-8lf%%\t%-14d %lf s",(mark_0-start_t)*ti,(mark_1-mark_0)*ti,(mark_2-mark_1)*ti,(mark_3-mark_2)*ti,(end_t-mark_3)*ti,(end_t-start_t)*ti,(EgZ-EgT)/sqrt((EgZ-EgT)*(EgZ-EgT)),((double)k/(double)nt)*100.0, np, ((end_t-t0)/((double)k/(double)ti))*(nt-k)/ti); fflush(stdout);}


	}


	if (my_rank == 0) printf("\n\nTime Elapsed: \n%lf s\n", (end_t-t0));



if (my_rank==0)
{
	netcdf_close(ncid0);
	netcdf_close(ncid1);
	netcdf_close(ncid2);
}


	/*Clearing up of memory*/
	free(n_data); free(n1_data); free(n2_data); free(n3_data);
	free(fn1_data); free(fn2_data); free(fn3_data);
	free(v_data); free(v1_data); free(v2_data); free(v3_data);
	free(fv1_data); free(fv2_data); free(fv3_data);
	free(phi_data);
	free(phi_k_data);
	free(nflux_data);

	free(nfsa); free(nfluxfsa); free(phifsa); free(vfsa);

	free(n); free(n1); free(n2); free(n3);
	free(fn1); free(fn2); free(fn3);
	free(v); free(v1); free(v2); free(v3);
	free(fv1); free(fv2); free(fv3);
	free(phi);
	free(phi_k);
	free(nflux);
	free(x); free(y); free(mm); free(nn);

	free(nprobes);	free(nfsaprobes);
	free(phiprobes);	free(phifsaprobes);
	free(vortprobes);	free(vortfsaprobes);

	free(sendbuf); free(recvbuf);
	free(sendbuf_data); free(recvbuf_data);

	free(startx); free(starty);
	free(endx); free(endy);
	free(lengthx); free(lengthy);


	/*Finalize MPI*/
	MPI_Finalize();

	exit(EXIT_SUCCESS);
}


/*	External function will setup the grid baed on the requested size.
	Two dimensional array is setup using malloc calls.
	This particular implementation outputs a 2D contiguous array */
char make_2dArray(int grid_size, double **grid, double *data)
{
	char success_flag;
	int i;

	success_flag=1;
	for(i=0; i<grid_size; i++)
	{
		grid[i] = data + i*grid_size;
		if(grid[i]==NULL)
		{
			fprintf(stderr, "****************************************\n");
			fprintf(stderr, "** Can't allocate memory for the grid **\n");
			fprintf(stderr, "****************************************\n");
			success_flag=0;
			break;
		}
	}

	return(success_flag);
}

double timer()
{
	double timing;

	timing = MPI_Wtime();

	return timing;
}

int* netcdf_creator(char* file_name, int ndims, int grid_size, int nt)
{
	/* IDs for the netCDF file, dimensions */
	int ncid, x_dimid, y_dimid, t_dimid;
	int dimids[3];

	/* Error handling. */
	int retval;

	/* Create the file. */
	if((retval = nc_create(file_name, NC_CLOBBER|NC_NETCDF4, &ncid)))
		ERR(retval);

	/* Define the dimensions. 3d or 1d. */
	if(ndims == 3)
	{
		if((retval = nc_def_dim(ncid, "x", grid_size, &x_dimid)))
			ERR(retval);
		if((retval = nc_def_dim(ncid, "y", grid_size, &y_dimid)))
			ERR(retval);

		if((retval = nc_def_dim(ncid, "t", nt, &t_dimid)))
			ERR(retval);

		dimids[0] = t_dimid;
		dimids[1] = x_dimid;
		dimids[2] = y_dimid;

	}
	else if(ndims == 1)
	{
		if((retval = nc_def_dim(ncid, "t", nt, &t_dimid)))
			ERR(retval);
		dimids[0] = t_dimid;
	}

	/* Markers to return. */
	int* markers;
	markers = (int*)malloc((1+ndims)*sizeof(int));

	markers[0] = ncid;
	if(ndims == 3)
	{
		markers[1] = dimids[0];
		markers[2] = dimids[1];
		markers[3] = dimids[2];
	}
	else if(ndims == 1)
	{
		markers[1] = dimids[0];
	}

	return markers;
}

int netcdf_meta(int ncid, int ndims, int* dimids, char* var)
{
	/* ID for the variable */
	int varid;

	/* Error handling. */
	int retval;

	/* Define the netCDF variable for data. */
	if ((retval = nc_def_var(ncid, var, NC_DOUBLE, ndims, dimids, &varid)))
		ERR(retval);

	return varid;
}

void netcdf_endmeta(int ncid)
{
	/* Error handling. */
	int retval;

	/* End define mode. */
	if ((retval = nc_enddef(ncid)))
		ERR(retval);

	return;
}

void netcdf_write1d(int ncid, int varid, double var, int k)
{
	size_t start[1];
	size_t count[1];

	start[0] = k;

	count[0] = 1;

	/* Error handling. */
	int retval;

	/* Write the data. */
	if((retval = nc_put_vara_double(ncid, varid, start, count, &var)))
		ERR(retval);

	return;
}

void netcdf_write2d(int ncid, int varid, double **var, int k, int grid_size)
{
	size_t start[3];
	size_t count[3];

	start[0] = k;
	start[1] = 0;
	start[2] = 0;

	count[0] = 1;
	count[1] = grid_size;
	count[2] = grid_size;

	/* Error handling. */
	int retval;

	/* Write the data. */
	if((retval = nc_put_vara_double(ncid, varid, start, count, var[0])))
		ERR(retval);

	return;
}

void netcdf_close(int ncid)
{
	/* Error handling. */
	int retval;

	/* Close the file.  */
	if((retval = nc_close(ncid)))
		ERR(retval);

	printf("\n *** SUCCESS writing file, %d!\n", ncid);

	return;
}

/*Sets up the ranges in x and y that each rank is responsible for traversing*/
/*2 columns are setup and they are split into rows according to the number of
  even numbered ranks(left column) and odd numbered ranks(right column).
  So p=3 would have 2 columns with the leftmost one being split into 2 rows
  each containing rank 0 and rank 2 with the singular element on the rightmost column
  being rank 1.
  p=4 would have 2 columns with 2 rows with rank 0 and rank 2 being in the leftmost column
  and rank 1 and rank 3 being in the rightmost column.
  p=5 would have 2 columns with 3 rows on the leftmost side containing ranks 0,2,4
  and with 2 rows on the rightmost side containing ranks 1,3.
  Remainders of splitting 256 by the number of rows or columns
  are attached to last rows or columns depending on the dimension in which there is 
  a remainder.*/
void coordinate_map(int **startx, int **starty, int **endx, int **endy, int **lengthx, int **lengthy, int p, int nl, int my_rank)
{
	int i, j;

	*startx = (int *)malloc(p*sizeof(int));
	*starty = (int *)malloc(p*sizeof(int));
	*endx = (int *)malloc(p*sizeof(int));
	*endy = (int *)malloc(p*sizeof(int));
	*lengthx = (int *)malloc(p*sizeof(int));
	*lengthy = (int *)malloc(p*sizeof(int));

	int nx1, nx2, ny;
	ny = 1;
	if (p>=2) {ny = 2;}
	nx1 = (int)floor((double)p/(double)ny);
	nx2 = (int)ceil((double)p/(double)ny);

	int splity, splitx1, splitx2;
	splity = (int)floor((double)nl/(double)ny);
	splitx1 = (int)floor((double)nl/(double)nx1);
	splitx2 = (int)floor((double)nl/(double)nx2);

	int remain_row, remain_col_1, remain_col_2;
	remain_row = (int)ceil((double)nl/(double)ny) - splity;
	remain_col_1 = (int)ceil((double)nl/(double)nx1)-splitx1;
	remain_col_2 = (int)ceil((double)nl/(double)nx2)-splitx2;

	for (i = 0; i < p; i++)
	{
		if (i%2 == 0)
		{
			(*startx)[i] = 1;
			(*endx)[i] = (*startx)[i] + splity;
			(*lengthx)[i] = (*endx)[i] - (*startx)[i];

			(*starty)[i] = (i/2)*splitx2 + 1;
			(*endy)[i] = (*starty)[i] + splitx2;
			(*lengthy)[i] = (*endy)[i] - (*starty)[i];
		}
		else if (i%2 == 1)
		{
			(*startx)[i] = splity + 1;
			(*endx)[i] = (*startx)[i] + splity;
			(*lengthx)[i] = (*endx)[i] - (*startx)[i];

			(*starty)[i] = ((i+1)/2 - 1)*splitx1 + 1;
			(*endy)[i] = (*starty)[i] + splitx1;
			(*lengthy)[i] = (*endy)[i] - (*starty)[i];
		}

		if (remain_row == 1)
		{
			if (i%2 == 1)
			{
				(*endx)[i] += 1;
				(*lengthx)[i] += 1;
			}
		}

	}

	if (remain_col_1 == 1)
	{
		j = 1;
		while (j < (p-1)) {j = j + 2;}
		(*endy)[j] += 1;
		(*lengthy)[j] += 1;
	}

	if (remain_col_2 == 1)
	{
		j = 0;
		while (j < (p-2)) {j = j + 2;}
		(*endy)[j] += 1;
		(*lengthy)[j] += 1;
	}

}

/*Simple function to allocate sendbuf and recvbuf contiguously*/
void sendrecv_alloc(double ***sendbuf, double ***recvbuf, double **sendbuf_data, double **recvbuf_data, int nl)
{
	char alloc_success;

	*sendbuf_data = (double *)malloc((nl+2)*(nl+2)*sizeof(double));
	if ((*sendbuf_data)==NULL) {printf("Allocation of sendbuf_data failed\n"); exit(EXIT_FAILURE);}

	*sendbuf = (double **)malloc((nl+2)*sizeof(double *));
	if ((*sendbuf)==NULL) {printf("Allocation of sendbuf failed\n"); exit(EXIT_FAILURE);}
	else alloc_success = make_2dArray(nl+2, (*sendbuf), (*sendbuf_data));
	if (!alloc_success) {printf("Allocation of sendbuf failed\n"); exit(EXIT_FAILURE);}


	*recvbuf_data = (double *)malloc((nl+2)*(nl+2)*sizeof(double));
	if ((*recvbuf_data)==NULL) {printf("Allocation of recvbuf_data failed\n"); exit(EXIT_FAILURE);}

	*recvbuf = (double **)malloc((nl+2)*sizeof(double *));
	if ((*recvbuf)==NULL) {printf("Allocation of recvbuf failed\n"); exit(EXIT_FAILURE);}
	else alloc_success = make_2dArray(nl+2, (*recvbuf), (*recvbuf_data));
	if (!alloc_success) {printf("Allocation of recvbuf failed\n"); exit(EXIT_FAILURE);}

}

/*Distributes the section of each split variable to rank 0(to then be broadcast at the start of the next iteration).*/
void distribute_to_root_rank(double ***data, int my_rank, double ***sendbuf, double ***recvbuf, int nl, int p, int **startx, int **starty, int **endx, int **endy)
{
	int i, j, k;
	MPI_Status status_info;

	if (my_rank != 0)
	{
		memcpy(&((*sendbuf)[0][0]), &((*data)[0][0]), (nl+2)*(nl+2)*sizeof(double));
		MPI_Send(&((*sendbuf)[0][0]), (nl+2)*(nl+2), MPI_DOUBLE, 0, my_rank*100, MPI_COMM_WORLD);
	}
	else
	{
		for (i = 1; i < p; i++)
		{
			memset(&((*recvbuf)[0][0]), 0, (nl+2)*(nl+2)*sizeof(double));
			MPI_Recv(&((*recvbuf)[0][0]), (nl+2)*(nl+2), MPI_DOUBLE, i, i*100, MPI_COMM_WORLD, &status_info);
			for (j = (*startx)[i]; j < (*endx)[i]; j++)
			{
				for (k = (*starty)[i]; k < (*endy)[i]; k++)
				{
					(*data)[j][k] = (*recvbuf)[j][k];
				}
			}
		}
	}
}
