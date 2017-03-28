/* 
Description:
============
Program to compute relative probabilities that each donor is good.

Uses Cuba to perform Monte Carlo integration (Suave method).


Inputs:
=======
Takes as input the prior shape parameters on phi, beta and epsilon, followed by the the current state.  If the current state is {(1, 2), (3, 4), (5, 6)}, where (s, f) represents successes and failures for a given donor, and phi_a = 20, phi_b = 30, eps_a = 40, eps_b = 50, beta_a = 60 and beta_b = 70, the inputs would be:
./ppp 20 30 40 50 60 70 1 2 3 4 5 6

Outputs:
========
Outputs the relative posterior donor goodness probabilities of each donor to STDOUT, i.e.
outputs p_i^2/sum_i p_i^2 for all donors i


Compilation:
============
Compile with :  g++ -o ppp -Wall ppp.c -lgsl -lcuba -lgslcblas -lm

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_integration.h>
#include "cuba.h"

/* Computes a linspace between a and b with 101 increments. */
double * linspace(double a, double b){
  static double array[101];
  double step = (b-a) / 100;
  int j;
  for (j = 0; j < 101; j++){
    array[j] = j * step;
  }
  return array;
}

/* Computes epsilon from beta and gamma */
double epsilon( double beta, double gamma ){
  double eps;
  eps = beta + gamma - beta*gamma;
  return eps;
}

static int Integrand(const int *ndim, const double xx[],
  const int *ncomp, double ff[], void *userdata) {

#define phi xx[0]
#define beta xx[1]
#define gamma xx[2]
#define f ff[0]

  /* Parse command line arguments corresponding to state Q.
     Arguments come as numbers which are read two by two as s_i, f_i pairs (successes and failures). 
  */  
  int* donordata = (int *)userdata;
  int ndonors = donordata[0];  
  double phi_a = (double)donordata[1];
  double phi_b = (double)donordata[2];
  double beta_a = (double)donordata[3];
  double beta_b = (double)donordata[4];
  double eps_a = (double)donordata[5];
  double eps_b = (double)donordata[6];
  int donor_index = donordata[7];
  int nstatevals = 8 + ndonors * 2;
  int successes[ndonors];
  int failures[ndonors];
  int j;
  int donorcounter = -1;

  for (j = 7; j < nstatevals; j++){
    if (j % 2 == 1){
      donorcounter++;
    }
    if (j % 2 == 1){
      successes[donorcounter] = donordata[j+1];
    }
    if (j % 2 == 0){
      failures[donorcounter] = donordata[j+1];
    }
  }


  /* Compute integrand */
  double eps_val;
  double product = 1;
  double inner_term = 0;
  int i;

  eps_val = epsilon(beta, gamma);

  /* Loop through other donors */
  if (donor_index == 0){
    for (i=1; i<ndonors; i++){
      inner_term = phi * pow(eps_val,(float)successes[i]) * pow((1-eps_val),(float)failures[i]) + (1-phi) * pow(beta,(float)successes[i]) * pow((1-beta),(float)failures[i]);
      product = product * inner_term; 
    }
    product = product * pow(eps_val,(float)successes[donor_index]) * pow((1-eps_val),(float)failures[donor_index]);
  }

  else if (donor_index > 0 && donor_index < ndonors-1){
    for (i=0; i<donor_index; i++){
      inner_term = phi * pow(eps_val,(float)successes[i]) * pow((1-eps_val),(float)failures[i]) + (1-phi) * pow(beta,(float)successes[i]) * pow((1-beta),(float)failures[i]);
      product = product * inner_term; 
    }
    for (i=donor_index+1; i<ndonors; i++){
      inner_term = phi * pow(eps_val,(float)successes[i]) * pow((1-eps_val),(float)failures[i]) + (1-phi) * pow(beta,(float)successes[i]) * pow((1-beta),(float)failures[i]);
      product = product * inner_term; 
    }
    product = product * pow(eps_val,(float)successes[donor_index]) * pow((1-eps_val),(float)failures[donor_index]);
  }

  else if (donor_index == ndonors-1){
    for (i=0; i<donor_index; i++){
      inner_term = phi * pow(eps_val,(float)successes[i]) * pow((1-eps_val),(float)failures[i]) + (1-phi) * pow(beta,(float)successes[i]) * pow((1-beta),(float)failures[i]);
      product = product * inner_term; 
    }
    product = product * pow(eps_val,(float)successes[donor_index]) * pow((1-eps_val),(float)failures[donor_index]);
  }

  f = product * pow(eps_val, eps_a) * pow((1-eps_val), eps_b) * pow(beta, beta_a) * pow((1-beta), beta_b) * pow(phi, phi_a) * pow((1-phi), phi_b);
  
  return 0;
}


#define NDIM 3
#define NCOMP 1
#define NVEC 1
#define EPSREL 1e-3
#define EPSABS 1e-12
#define VERBOSE 0
#define LAST 4
#define SEED 0
#define MINEVAL 0
#define MAXEVAL 1000000

#define NSTART 1000
#define NINCREASE 500
#define NBATCH 1000
#define GRIDNO 0
#define STATEFILE NULL
#define SPIN NULL

#define NNEW 1000
#define NMIN 2
#define FLATNESS 25.

#define KEY1 47
#define KEY2 1
#define KEY3 1
#define MAXPASS 5
#define BORDER 0.
#define MAXCHISQ 10.
#define MINDEVIATION .25
#define NGIVEN 0
#define LDXGIVEN NDIM
#define NEXTRA 0

#define KEY 0

int main( int argc, char *argv[] ) {
  
  /* Parse command line arguments corresponding to state Q.
     Inputs: phi_shape_parameter_a, phi_shape_parameter_b, beta_shape_parameter_a, beta_shape_parameter_b, followed
     by the state successes and failures, which are read two by two as s_i, f_i pairs. 
  */
  int i;
  int ndonors = (argc - 1 - 6)/2;
  double donor_goodness_probs[ndonors]; /* Posterior probabilities of donors being good (relative to each other) */
  int arrlen = argc;
  int STATEDATA[arrlen + 1];  /* current state, donor i: (si, fi), plus donor index (in the last entry) */
  void * USERDATA;
  STATEDATA[0] = ndonors;
  /* Priors */
  for (i=1; i<7; i++){
    STATEDATA[i] = atoi(argv[i]);
  }
  /* Donor index */
  STATEDATA[7] = 0;
  /* Successes and failures */
  for (i=8; i<arrlen+1; i++){
    STATEDATA[i] = atoi(argv[i-1]);
  } 

  int nregions, neval, fail;
  double integral[NCOMP], error[NCOMP], prob[NCOMP];
  double donor_goodness_prob;

  for (i = 0; i < ndonors; i++){

    STATEDATA[7] = i;

    /* Compute goodness probability */
    USERDATA = &STATEDATA;
    Suave(NDIM, NCOMP, Integrand, USERDATA, NVEC,
	  EPSREL, EPSABS, VERBOSE | LAST, SEED,
	  MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,
	  STATEFILE, SPIN,
	  &nregions, &neval, &fail, integral, error, prob);
    donor_goodness_prob = integral[0];

    /* Compute P(sigma_i | X) as ratio of Q(s_i + 1, f_i) and Q(s_i, f_i) */
    donor_goodness_probs[i] = donor_goodness_prob;
       
  }

  /* Normalize donor goodness probabilities */
  double norm_term = 0;
  for (i=0; i<ndonors; i++){
    norm_term = norm_term + pow(donor_goodness_probs[i], 2);
  }
  for (i=0; i<ndonors; i++){
    donor_goodness_probs[i] = pow(donor_goodness_probs[i], 2) / norm_term;
  }

  /* Print output */
  for (i = 0; i < ndonors; i++){
    printf("%.8f\n", donor_goodness_probs[i]);
  }

  return 0;

}

