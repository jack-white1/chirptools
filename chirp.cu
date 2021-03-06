// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdbool.h>

// includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

// includes fftw
#include <fftw3.h>

#define _USE_MATH_DEFINES

//#define SIGLEN 20000

struct signalParams{
  double sampleRate;
  unsigned long int signalLength;
  char zeroBinAverageFlag;
};

struct chirpParams {
  char chirpType; //P = polynomial sine, E = exponential sine, H = hyperbolic sine, R = pulse polynomial, I = integer pulse, C = custom
  double startingPhase; //radians, usually 0
  double amplitude; //V
  double startingFreq; //Hz
  double endingFreq; //Hz
  double startTime; //seconds = starting point of chirp in signal
  double chirpTime; //seconds = length of time over which frequency sweep occurs
  double pExponent; //exponent of polynomial (eg 1=linear, 2=quadratic, cubic) chirp
  double exponentialBase; //base which is raised to t in exponential chirp
  double f0Hyp; //f0 in hyperbolic chirp instantaneous frequency (f0*f1*T/((f0-f1)*t+f1*T))
  double f1Hyp; //f1 in hyperbolic chirp instantaneous frequency
  double hypT; //T in hyperbolic chirp instantaneous frequency
  double awgnStDev; //V, standard deviation of AWGN
  double sqDutyCycle; //fractional ON time of square pulse chirp
  long int startingIntegerPeriodSamples; //number of samples for first iteration of integer pulse
  long int integerPulseLengthSamples; //number of samples for length of on pulse
};

struct noiseParams {
  //char noiseColor; //W = white, P = pink, R = red/brown, B = blue, V = violet
  long unsigned int noiseLength; //length of noise signal
  double2* noisePtr; //memory location of noise array
  double cutoffHigh;
  double cutoffLow;
  double beta; //1 for pink noise, 0 for white noise
  double noiseMultiplier; //multiplies random noise by this values
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }  //from https://stackoverflow.com/a/14038590 to avoid the whole cudaError_t e = cudaMemCpy() stuff, makes code more compact

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)  //more code from stack overflow /a/14038590 to do inline errorstring print
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

void manual_C2Cinv_fftw(double2* inputData, double2* outputData, long unsigned int inputLength){
  fftw_complex *in, *out;
  fftw_plan p;

  in  = (fftw_complex*) inputData;
  out = (fftw_complex*) outputData;

  p = fftw_plan_dft_1d(inputLength, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

  fftw_execute(p);

  fftw_destroy_plan(p);
  //fftw_free(in);
  //fftw_free(out);

}

void manual_C2Cinv_cuda(double2* h_inputData, double2* h_outputData, long unsigned int inputLength){
  // Allocate memory
  size_t GPU_input_size = inputLength*sizeof(double2);
  size_t GPU_output_size = inputLength*sizeof(double2);

  double2* d_input;
  double2* d_output;
  gpuErrchk(cudaMalloc((void **) &d_input,  GPU_input_size));
  gpuErrchk(cudaMalloc((void **) &d_output,  GPU_output_size));

  // Transfer input to GPU
  cudaMemcpy(d_input, h_inputData, GPU_input_size, cudaMemcpyHostToDevice);

  // Execute FFT using cuFFT
  cufftHandle plan;
  cufftPlan1d(&plan, inputLength, CUFFT_Z2Z, 1);
  cufftExecZ2Z(plan, d_input, d_output, CUFFT_INVERSE);
  cufftDestroy(plan);

  cudaMemcpy((float *) h_outputData, d_output, GPU_output_size, cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}

void complexParameterisedNoiseGenerator(struct noiseParams noise, struct signalParams signalParams){
  //using algorithm from "On Generating Power Law Noise", Timmer + Konig (1995)
  //1. choose a power law spectrum S(omega) approx= (1/omega)^beta
  //2. For each Fourier frequency omega_i, draw two gaussian distributed random numbers,
  //   multiply them by sqrt(0.5*S(omega_i)) approx= (1/omega)^(beta/2), use these two numbers
  //   as the real and imaginary part of the fourier transform of the desired data
  //3  In the case of an even number of data points, f(omega_nyquist) is always real so only one number has to be drawn
  //4  To obtain a real valued time series, choose the Fourier compenents for the negative frequencies
  //   according to f(-omega_i)=f*(omega_i) where the asterisk denotes complex conjugation
  //5  Obtain the time series by IFFT f(omega)

  long unsigned int spectrumLength = noise.noiseLength;//always an even number
  double2 spectrum[signalParams.signalLength];
  double U1, U2, Z0, Z1, omega_i;
  double bandwidth = noise.cutoffHigh - noise.cutoffLow;
  double reComponent, imComponent;

  spectrum[0].x = 0.0;
  spectrum[0].y = 0.0;

  double reTotal = 0.0;
  double imTotal = 0.0;
  for (long unsigned int i = 1; i < signalParams.signalLength/2 + 1; i++){
    omega_i = i*signalParams.sampleRate/signalParams.signalLength;
    U1 = noise.noiseMultiplier*rand() / (double)RAND_MAX;
    U2 = noise.noiseMultiplier*rand() / (double)RAND_MAX;
    Z0 = sqrt(-2*log(U1))*cos(2*M_PI*U2); //Box-Muller transform
    Z1 = sqrt(-2*log(U1))*sin(2*M_PI*U2); //plotted Z0, Z1 they are verified to be Gaussian
    reComponent = Z0*sqrt(0.5*pow(1/omega_i,noise.beta/2));
    imComponent = Z1*sqrt(0.5*pow(1/omega_i,noise.beta/2));
    spectrum[i].x = reComponent;
    spectrum[i].y = imComponent;
    reTotal = reTotal + reComponent;
    imTotal = imTotal + imComponent;
    //printf("%lf, %lf\n", spectrum[i].x, spectrum[i].y);
    //printf("%lf, %lf\n", Z0, Z1);
  }

  for (long unsigned int i = 1; i < signalParams.signalLength/2; i++){
    spectrum[signalParams.signalLength/2+i].x = spectrum[signalParams.signalLength/2-i].x;
    spectrum[signalParams.signalLength/2+i].y = -spectrum[signalParams.signalLength/2-i].y;
  }
  /*
  if (signalParams.zeroBinAverageFlag == 1){
    spectrum[0].x = reTotal/(signalParams.signalLength-1);
    spectrum[0].y = imTotal/(signalParams.signalLength-1);
    printf("0th bin: %lf, %lf\n", spectrum[0].x, spectrum[0].y);
  }
  */
  
  if (signalParams.zeroBinAverageFlag == 1){
    U1 = noise.noiseMultiplier*rand() / (double)RAND_MAX;
    U2 = noise.noiseMultiplier*rand() / (double)RAND_MAX;
    Z0 = sqrt(-2*log(U1))*cos(2*M_PI*U2); //Box-Muller transform
    Z1 = sqrt(-2*log(U1))*sin(2*M_PI*U2); //plotted Z0, Z1 they are verified to be Gaussian
    spectrum[0].x = Z0;
    spectrum[0].y = Z1;
    //printf("0th bin: %lf, %lf\n", spectrum[0].x, spectrum[0].y);
  }
  
  
  
  manual_C2Cinv_fftw(spectrum, noise.noisePtr, spectrumLength);
  //manual_C2Cinv_cuda(spectrum, noise.noisePtr, spectrumLength);


}

void takeRealPart(double2* inputPtr, double* outputPtr, struct signalParams params){
  for (unsigned int i = 0; i < params.signalLength; i++){
    outputPtr[i] = inputPtr[i].x;
  }

}

void combineSignalAndNoise(double* signalPtr, double* noisePtr, double* outputPtr, struct signalParams params){
  for (unsigned int i = 0; i < params.signalLength; i++){
    outputPtr[i] = signalPtr[i] + noisePtr[i];
  }
}


double AWGN_generator(double stdev){
  // modified from Dr Cagri Tanriover's code from https://www.embeddedrelated.com/showcode/311.php
  // Generates additive white Gaussian Noise samples with zero mean and a standard deviation of stdev
  double temp1;
  double temp2;
  double result;
  int p;

  p = 1;

  while( p > 0 ) {
    temp2 = ( rand() / ( (double)RAND_MAX ) ); //  rand() function generates an integer between 0 and  RAND_MAX, which is defined in stdlib.h.
    if ( temp2 == 0 ){
      p = 1;
    } else {
      p = -1;
    }
  }

  temp1 = cos( ( 2.0 * (double)M_PI ) * rand() / ( (double)RAND_MAX ) );
  result = sqrt( -2.0 * log( temp2 ) ) * temp1;

  return stdev*result;
}

double returnPolynomialSample(double timeStamp, struct chirpParams chirp) {
  if ((timeStamp >= chirp.startTime) && (timeStamp <= chirp.startTime + chirp.chirpTime)){
    double t = timeStamp-chirp.startTime;
    double chirpRate = (chirp.endingFreq-chirp.startingFreq)/chirp.chirpTime;
    double sinArg = chirp.startingPhase + 2*M_PI*t*(chirpRate*pow(t,chirp.pExponent)/2 + chirp.startingFreq*t);
    double output = chirp.amplitude*sin(sinArg);
    //printf("Returning: %lf\n", output);
    return output;
  } else {
    return 0;
  }
}

double returnExponentialSample(double timeStamp, struct chirpParams chirp) {
  if ((timeStamp >= chirp.startTime) && (timeStamp <= chirp.startTime + chirp.chirpTime)){
    double t = timeStamp-chirp.startTime;
    double sinArg = chirp.startingPhase + 2*M_PI*chirp.startingFreq*((pow(chirp.exponentialBase,t)-1)/log(chirp.exponentialBase));
    double output = chirp.amplitude*sin(sinArg);
    return output;
  } else {
    return 0;
  }
}

double returnHyperbolicSample(double timeStamp, struct chirpParams chirp) {
  if ((timeStamp >= chirp.startTime) && (timeStamp <= chirp.startTime + chirp.chirpTime)){
    double t = timeStamp-chirp.startTime;
    double f0 = chirp.f0Hyp;
    double f1 = chirp.f1Hyp;
    double T = chirp.hypT;
    double sinArg = chirp.startingPhase - 2*M_PI*f0*f1*T*log(1-t*(f1-f0)/(f1*T))/(f1-f0);
    double output = chirp.amplitude*sin(sinArg);
    return output;
  } else {
    return 0;
  }
}

double squarewave(double arg, double dutyCycle){ //dutyCycle is fractional (0..1) ON time throughout 0->2PI
 //if ((arg % (2.0*M_PI)) >= (2*M_PI*(1-dutyCycle))){
 if (fmod(arg,2*M_PI) >= 2*M_PI*(1-dutyCycle)){
 return 1.0; //case ON
    } else {
   return 0.0; //case OFF
  }
}

double returnPulsePolynomialChirpSample(double timeStamp, struct chirpParams chirp){
  if ((timeStamp >= chirp.startTime) && (timeStamp <= chirp.startTime + chirp.chirpTime)){
    double t = timeStamp-chirp.startTime;
    double chirpRate = (chirp.endingFreq-chirp.startingFreq)/chirp.chirpTime;
    double sqArg = chirp.startingPhase + 2*M_PI*t*(chirpRate*pow(t,chirp.pExponent)/2 + chirp.startingFreq*t);
    double output = chirp.amplitude*squarewave(sqArg,chirp.sqDutyCycle)*AWGN_generator(chirp.awgnStDev);
    return output;
  } else {
    return 0;
  }
}

long int calcOnIndexesIntegerPulse(long int* onIndicesArr, double sampleRate, double timeStamp, struct chirpParams chirp ){
  double signalLength = chirp.chirpTime/sampleRate;
  double startPeriod = (double) chirp.startingIntegerPeriodSamples;
  double pulseLength = (double) chirp.integerPulseLengthSamples;
  double quadTermA = startPeriod + pulseLength + 0.5;
  double quadTermB = 2*signalLength;
  double nWindowsRootA = -quadTermA + sqrt(quadTermA*quadTermA - quadTermB);
  double nWindowsRootB = -quadTermA - sqrt(quadTermA*quadTermA - quadTermB);
  double nWindows = (((nWindowsRootA) >= (nWindowsRootB)) ? (nWindowsRootA) : (nWindowsRootB));
  long int nWindowsInt = (long int) (nWindows + 0.5);
   /*[LB1,UB1,LB2,UB2...]*/
  onIndicesArr = (long int*)malloc(2*nWindowsInt*sizeof(long int));
  long int chirpLengthSamples = chirp.chirpTime*sampleRate;
  long int indicesArrIdx = 0;
//    long int signalIdx = 0;
  long int windowCount = 0;
  while(indicesArrIdx < chirpLengthSamples){
    indicesArrIdx = indicesArrIdx + chirp.startingIntegerPeriodSamples - windowCount;
    long int LB = indicesArrIdx;
    indicesArrIdx = indicesArrIdx + chirp.integerPulseLengthSamples;
    long int UB = indicesArrIdx;
    onIndicesArr[indicesArrIdx] = LB;
    onIndicesArr[indicesArrIdx+1] = UB;
    indicesArrIdx = indicesArrIdx + 2;
    windowCount = windowCount + 1;
  }
  return windowCount;
}

double returnIntegerPulseChirpSample(long int globalSampleIndex, long int nWindowsInt, long int* onIndicesArr, double timeStamp, struct chirpParams chirp){
  if ((timeStamp >= chirp.startTime) && (timeStamp <= chirp.startTime + chirp.chirpTime)){
    double sampleRate = globalSampleIndex/timeStamp;
    long int startSampleIndex = chirp.startTime*sampleRate;
    long int localSampleIndex = globalSampleIndex - startSampleIndex;
    double output = 0;
    for (int i = 0; i < nWindowsInt; i = i + 2){
      if ((localSampleIndex >= onIndicesArr[i]) && (localSampleIndex <= onIndicesArr[i+1])){
        output = 1;
      }
    }
    /*

    */
    return output;
  } else {
    return 0;
  }
}

double returnSample(long int sampleIndex, double sampleRate, struct chirpParams chirp){
  double timeStamp = sampleIndex/sampleRate;
  if (chirp.chirpType == 'P'){
    return returnPolynomialSample(timeStamp, chirp);
  } else if (chirp.chirpType == 'E'){
    return returnExponentialSample(timeStamp, chirp);
  } else if (chirp.chirpType == 'H'){
    return returnHyperbolicSample(timeStamp, chirp);
  } else if (chirp.chirpType == 'R'){
    return returnPulsePolynomialChirpSample(timeStamp, chirp);
  } else if (chirp.chirpType == 'I'){
    long int* onIndicesArr;
    long int nWindowsInt = calcOnIndexesIntegerPulse(onIndicesArr, sampleRate, timeStamp, chirp);
    return returnIntegerPulseChirpSample(sampleIndex, nWindowsInt, onIndicesArr, timeStamp, chirp);
  } else {
    return 0.0; //TODO PROPER DEFAULT CASE
  }
}


void setDefaultChirpNoiseParams(chirpParams* targetChirp, noiseParams* targetNoise){
  targetChirp->chirpType        = 'P';
  targetChirp->startingPhase    = 0.0;
  targetChirp->startingFreq     = 100;
  targetChirp->endingFreq       = 1000;
  targetChirp->startTime        = 1.0;
  targetChirp->chirpTime        = 10.0;
  targetChirp->amplitude        = 1000.0;
  targetChirp->pExponent        = 1.0;
  targetChirp->sqDutyCycle      = 0.1;
  targetChirp->awgnStDev        = 1.0;
  targetNoise->beta             = 1.0;
  targetNoise->cutoffLow        = 1.0;
  targetNoise->cutoffHigh       = 10.0;
  targetNoise->noiseMultiplier  = 1.0;
  return;
}

void setCustomChirpNoiseParams( char* argv[], chirpParams* targetChirp, noiseParams* targetNoise){
  targetChirp->chirpType                      = (char) *argv[1]; //default P
  targetChirp->startingPhase                  = atof(argv[2]);  //default 0.0
  targetChirp->startingFreq                   = atof(argv[3]);  //default 100.0
  targetChirp->endingFreq                     = atof(argv[4]);  //default 1000.0
  targetChirp->startTime                      = atof(argv[5]);  //default 1.0
  targetChirp->chirpTime                      = atof(argv[6]);  //default 10.0
  targetChirp->amplitude                      = atof(argv[7]);  //default 1000.0
  targetChirp->pExponent                      = atof(argv[8]);  //default 1
  targetChirp->sqDutyCycle                    = atof(argv[9]);  //default 0.1
  targetChirp->awgnStDev                      = atof(argv[10]); //default 1
  targetChirp->startingIntegerPeriodSamples   = atof(argv[11]); //number of samples for first iteration of integer pulse
  targetChirp->integerPulseLengthSamples      = atof(argv[12]); //number of samples for length of on pulse
  targetNoise->beta                           = atof(argv[13]); //default 1
  targetNoise->cutoffLow                      = atof(argv[14]); //default 1
  targetNoise->cutoffHigh                     = atof(argv[15]); //default 10
  targetNoise->noiseMultiplier                = atof(argv[16]); // default 1
  return;
}

void printChirpParams(chirpParams* targetChirp){
  printf("chirp.chirpType: %c\n",              targetChirp->chirpType);
  printf("chirp.startingPhase: %lf radians\n", targetChirp->startingPhase);
  printf("chirp.startingFreq: %lf Hz\n",       targetChirp->startingFreq);
  printf("chirp.endingFreq: %lf Hz\n",         targetChirp->endingFreq);
  printf("chirp.startTime: %lf s\n",           targetChirp->startTime);
  printf("chirp.chirpTime: %lf s\n",           targetChirp->chirpTime);
  printf("chirp.amplitude: %lf\n",             targetChirp->amplitude);
  printf("chirp.pExponent: %lf\n",             targetChirp->pExponent);
  printf("chirp1.sqDutyCycle: %lf\n",          targetChirp->sqDutyCycle);
  printf("chirp1.awgnStDev: %lf\n",            targetChirp->awgnStDev);
}

void printNoiseParams(noiseParams* targetNoise){
  printf("noise.beta: %lf\n",                  targetNoise->beta);
  printf("noise.cutoffLow: %lf Hz\n",          targetNoise->cutoffLow);
  printf("noise.cutoffHigh: %lf Hz\n",         targetNoise->cutoffHigh);
  printf("noise.noiseMultiplier: %lf\n",       targetNoise->noiseMultiplier);
}

int main(int argc, char *argv[]){
  //srand(time(0));

  struct signalParams sample1;
  struct chirpParams chirp1;
  struct noiseParams noise1;

  if (argc == 15){
    setCustomChirpNoiseParams(argv, &chirp1, &noise1);
  } else if (argc == 1){
    setDefaultChirpNoiseParams(&chirp1, &noise1);
  } else {
    printf("ERROR wrong number of input args (need 15)\n");
  }

  sample1.sampleRate   = 8000;
  sample1.signalLength = 20000;
  sample1.zeroBinAverageFlag = 1;

  
  noise1.noiseLength     = sample1.signalLength;

  printf("noise.beta: %lf\n",                  noise1.beta);
  printf("noise.cutoffLow: %lf Hz\n",          noise1.cutoffLow);
  printf("noise.cutoffHigh: %lf Hz\n",         noise1.cutoffHigh);
  printf("noise.noiseMultiplier: %lf\n",       noise1.noiseMultiplier);

  printChirpParams(&chirp1);
  printNoiseParams(&noise1);

  double* signal;
  signal =(double*) malloc(sample1.signalLength*sizeof(double));
  memset((void*)signal, 0, sample1.signalLength*sizeof(double));

  double2* noiseArr;
  noiseArr = (double2*)malloc(sample1.signalLength*sizeof(double2));
  memset((void*)noiseArr, 0, sample1.signalLength*sizeof(double));

  noise1.noisePtr = noiseArr;
  
  if(noise1.noiseMultiplier != 0.0){
    complexParameterisedNoiseGenerator(noise1,sample1);
  }
  
  for (long int i = 0; i<sample1.signalLength; i++){
    signal[i] = returnSample(i,sample1.sampleRate,chirp1);
    //printf("%lf\n", signal[i]);
    //printf("noise1.noisePtr[%d].x = %lf\n", i, noise1.noisePtr[i].x);
    //printf("noise1.noisePtr[%d].y = %lf\n", i, noise1.noisePtr[i].y);
    //printf("%lf\n", noise1.noisePtr[i].x);
    //printf("%lf\n", noise1.noisePtr[i].y);
  }

  double* realNoise;
  realNoise = (double*) malloc(sample1.signalLength*sizeof(double));
  takeRealPart(noise1.noisePtr, realNoise, sample1);
  
  double* combinedSignal;
  combinedSignal = (double*) malloc(sample1.signalLength*sizeof(double));
  combineSignalAndNoise(signal, realNoise, combinedSignal, sample1);


  FILE *fptr;
  fptr = fopen("chirp.txt","w");

  if(fptr == NULL){
    printf("Error!");
    exit(1);
  }

  for (int i = 0; i<sample1.signalLength; i++){
    //printf("%lf\n", combinedSignal[i]);
    fprintf(fptr,"%lf\n",combinedSignal[i]);
  }
  fclose(fptr);

  printf("written successfully to chirp.txt\n");

  free(signal);
  free(noiseArr);
  free(realNoise);
  free(combinedSignal);

  return 0;
}


/*
  1. simulate different noise types.
  2. also think about how a user might "inject" their own noise into the code.
  3. Different pulse profiles.
  4. User's own pulse profile.
  5. Add interference (that is beyond a standard noise profile, by this I mean an interfering signal or another signal present, could be periodic, might not be, could be broadband, narrowband etc).
  6. Unit tests.
  7. Code review.
  8. Package it and make available on git (with build system etc).
  9. Have an interface so it can be called by another code or run standalone.
  10. If it's single frequency channel that make it multiple.
*/
