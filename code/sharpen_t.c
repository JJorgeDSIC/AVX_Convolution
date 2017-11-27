#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>



typedef double FLOAT;
//typedef float FLOAT;

// Cycle Counter Code
//
// Can be replaced with ippGetCpuFreqMhz and ippGetCpuClocks
// when IPP core functions are available.
//
typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;





// PPM Edge Enhancement Code
//
UINT8 header[22];
UINT8 R[76800];
UINT8 G[76800];
UINT8 B[76800];
UINT8 convR[76800];
UINT8 convG[76800];
UINT8 convB[76800];

#define K 4.0

FLOAT PSF[9] = {-K/8.0, -K/8.0, -K/8.0, -K/8.0, K+1.0, -K/8.0, -K/8.0, -K/8.0, -K/8.0};

int main(int argc, char *argv[])
{
    int fdin, fdout, bytesRead=0, bytesLeft, i, j,vuelta;
    double elapsed0, ucpu0, scpu0;
    double elapsed1, ucpu1, scpu1;
    FLOAT temp;


  

    if(argc < 2)
    {
       printf("Usage: sharpen file.ppm\n");
       exit(-1);
    }
    else
    {


        if((fdin = open(argv[1], O_RDONLY, 0644)) < 0)
        {
            printf("Error opening %s\n", argv[1]);
        }


        if((fdout = open("sharpen.ppm", (O_RDWR | O_CREAT), 0666)) < 0)
        {
            printf("Error opening %s\n", argv[1]);
        }

    }

    bytesLeft=21;

    //printf("Reading header\n");

    do
    {
        //printf("bytesRead=%d, bytesLeft=%d\n", bytesRead, bytesLeft);
        bytesRead=read(fdin, (void *)header, bytesLeft);
        bytesLeft -= bytesRead;
    } while(bytesLeft > 0);

    header[21]='\0';

    //printf("header = %s\n", header); 

    // Read RGB data
    for(i=0; i<76800; i++)
    {
        read(fdin, (void *)&R[i], 1); convR[i]=R[i];
        read(fdin, (void *)&G[i], 1); convG[i]=G[i];
        read(fdin, (void *)&B[i], 1); convB[i]=B[i];
    }

    // Start of convolution time stamp
    ctimer_(&elapsed0, &ucpu0, &scpu0);
    for (vuelta=1;vuelta<500;vuelta++)
    {

    // Skip first and last row, no neighbors to convolve with
    for(i=1; i<239; i++)
    {

        // Skip first and last column, no neighbors to convolve with
        for(j=1; j<319; j++)
        {
            temp=0;
            temp += (PSF[0] * (FLOAT)R[((i-1)*320)+j-1]);
            temp += (PSF[1] * (FLOAT)R[((i-1)*320)+j]);
            temp += (PSF[2] * (FLOAT)R[((i-1)*320)+j+1]);
            temp += (PSF[3] * (FLOAT)R[((i)*320)+j-1]);
            temp += (PSF[4] * (FLOAT)R[((i)*320)+j]);
            temp += (PSF[5] * (FLOAT)R[((i)*320)+j+1]);
            temp += (PSF[6] * (FLOAT)R[((i+1)*320)+j-1]);
            temp += (PSF[7] * (FLOAT)R[((i+1)*320)+j]);
            temp += (PSF[8] * (FLOAT)R[((i+1)*320)+j+1]);
	    if(temp<0.0) temp=0.0;
	    if(temp>255.0) temp=255.0;
	    convR[(i*320)+j]=(UINT8)temp;

            temp=0;
            temp += (PSF[0] * (FLOAT)G[((i-1)*320)+j-1]);
            temp += (PSF[1] * (FLOAT)G[((i-1)*320)+j]);
            temp += (PSF[2] * (FLOAT)G[((i-1)*320)+j+1]);
            temp += (PSF[3] * (FLOAT)G[((i)*320)+j-1]);
            temp += (PSF[4] * (FLOAT)G[((i)*320)+j]);
            temp += (PSF[5] * (FLOAT)G[((i)*320)+j+1]);
            temp += (PSF[6] * (FLOAT)G[((i+1)*320)+j-1]);
            temp += (PSF[7] * (FLOAT)G[((i+1)*320)+j]);
            temp += (PSF[8] * (FLOAT)G[((i+1)*320)+j+1]);
	    if(temp<0.0) temp=0.0;
	    if(temp>255.0) temp=255.0;
	    convG[(i*320)+j]=(UINT8)temp;

            temp=0;
            temp += (PSF[0] * (FLOAT)B[((i-1)*320)+j-1]);
            temp += (PSF[1] * (FLOAT)B[((i-1)*320)+j]);
            temp += (PSF[2] * (FLOAT)B[((i-1)*320)+j+1]);
            temp += (PSF[3] * (FLOAT)B[((i)*320)+j-1]);
            temp += (PSF[4] * (FLOAT)B[((i)*320)+j]);
            temp += (PSF[5] * (FLOAT)B[((i)*320)+j+1]);
            temp += (PSF[6] * (FLOAT)B[((i+1)*320)+j-1]);
            temp += (PSF[7] * (FLOAT)B[((i+1)*320)+j]);
            temp += (PSF[8] * (FLOAT)B[((i+1)*320)+j+1]);
	    if(temp<0.0) temp=0.0;
	    if(temp>255.0) temp=255.0;
	    convB[(i*320)+j]=(UINT8)temp;
        }
    }
}

    // End of convolution time stamp
 ctimer_(&elapsed1, &ucpu1, &scpu1);
    printf("Tiempo: %fs (real) %fs (cpu) %fs (sys)\n", 
                 elapsed1-elapsed0, ucpu1-ucpu0, scpu1-scpu0);

    write(fdout, (void *)header, 21);

    // Write RGB data
    for(i=0; i<76800; i++)
    {
        write(fdout, (void *)&convR[i], 1);
        write(fdout, (void *)&convG[i], 1);
        write(fdout, (void *)&convB[i], 1);
    }


    close(fdin);
    close(fdout);
 
}
