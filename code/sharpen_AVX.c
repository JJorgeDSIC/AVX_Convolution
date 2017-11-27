#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <immintrin.h>

typedef float FLOAT;
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
FLOAT fR[76800];
FLOAT fG[76800];
FLOAT fB[76800];

UINT8 convR[76800];
UINT8 convG[76800];
UINT8 convB[76800];

#define K 4.0

FLOAT PSF[9] = {-K/8.0, -K/8.0, -K/8.0, -K/8.0, K+1.0, -K/8.0, -K/8.0, -K/8.0, -K/8.0};

int main(int argc, char *argv[])
{
    int fdin, fdout, bytesRead=0, bytesLeft, i, j, t, vuelta;
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


    FLOAT kf = ((-K/8.0));
    FLOAT compes_value = (K/8.0) + (K+1.0);

    __m256* ptr_1;
    __m256* ptr_2;
    __m256* ptr_3;
    __m256 resTR, resMR, resBR, resTot, resValue;
    __m256 kv;
    __m256 ck;

    kv = _mm256_set1_ps(kf);
    ck = _mm256_set1_ps(compes_value);

    __m256 _max_mask = _mm256_set1_ps(255.0);
    __m256 _min_mask = _mm256_set1_ps(0.0);

    // Start of convolution time stamp
    ctimer_(&elapsed0, &ucpu0, &scpu0);

    for (vuelta=1;vuelta<500;vuelta++)
    {
		for(i=0; i<76800; i++)
		{
		    fR[i]=(FLOAT)R[i];
		    fG[i]=(FLOAT)G[i];
		    fB[i]=(FLOAT)B[i];
			
		}
		// Skip first and last row, no neighbors to convolve with
		//#pragma omp parallel for private(j,resTR, resMR, resBR, resTot, resValue, ptr_1, ptr_2, ptr_3)
		for(i=1; i<239; i++)
		{

			// Skip first and last column, no neighbors to convolve with
			for(j=1; j<312; j+=8)
			{
				//printf("J: %ds\n",j);

				//Top row
				ptr_1=(__m256*)&fR[((i-1)*320)+j-1];
				ptr_2=(__m256*)&fR[((i-1)*320)+j];
				ptr_3=(__m256*)&fR[((i-1)*320)+j+1];
				resTR = _mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));
				

				//Midle row
				ptr_1=(__m256*)&fR[((i)*320)+j-1];
				ptr_2=(__m256*)&fR[((i)*320)+j];
				ptr_3=(__m256*)&fR[((i)*320)+j+1];
				resMR=_mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));
				
				//ptr_1=(__m256*)&fR[((i)*320)+j];
				resValue=_mm256_mul_ps(*ptr_2,ck);

				//Bottom row
				ptr_1=(__m256*)&fR[((i+1)*320)+j-1];
				ptr_2=(__m256*)&fR[((i+1)*320)+j];
				ptr_3=(__m256*)&fR[((i+1)*320)+j+1];
				resBR=_mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));

				resTot=_mm256_add_ps(resBR, (_mm256_add_ps(resTR, resMR)));
				resTot=_mm256_mul_ps(resTot,kv);

				resTot=_mm256_add_ps(resTot,resValue);

				resTot = _mm256_max_ps(_min_mask, _mm256_min_ps(_max_mask, resTot));

				convR[(i*320)+j]=(UINT8)resTot[0];
				convR[(i*320)+j+1]=(UINT8)resTot[1];
				convR[(i*320)+j+2]=(UINT8)resTot[2];
				convR[(i*320)+j+3]=(UINT8)resTot[3];
				convR[(i*320)+j+4]=(UINT8)resTot[4];
				convR[(i*320)+j+5]=(UINT8)resTot[5];	
				convR[(i*320)+j+6]=(UINT8)resTot[6];	
				convR[(i*320)+j+7]=(UINT8)resTot[7];	
				/////////////////////////////////////

				//Top row
				ptr_1=(__m256*)&fG[((i-1)*320)+j-1];
				ptr_2=(__m256*)&fG[((i-1)*320)+j];
				ptr_3=(__m256*)&fG[((i-1)*320)+j+1];
				resTR = _mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));
				
				//Midle row
				ptr_1=(__m256*)&fG[((i)*320)+j-1];
				ptr_2=(__m256*)&fG[((i)*320)+j];
				ptr_3=(__m256*)&fG[((i)*320)+j+1];
				resMR=_mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));
				
				//Midle row - Pixels of Interest
				resValue=_mm256_mul_ps(*ptr_2,ck);

				//Bottom row
				ptr_1=(__m256*)&fG[((i+1)*320)+j-1];
				ptr_2=(__m256*)&fG[((i+1)*320)+j];
				ptr_3=(__m256*)&fG[((i+1)*320)+j+1];
				resBR=_mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));
		
				resTot=_mm256_add_ps(resBR, (_mm256_add_ps(resTR, resMR)));

				resTot=_mm256_mul_ps(resTot,kv);

				resTot=_mm256_add_ps(resTot,resValue);

				resTot = _mm256_max_ps(_min_mask, _mm256_min_ps(_max_mask, resTot));

				convG[(i*320)+j]=(UINT8)resTot[0];
				convG[(i*320)+j+1]=(UINT8)resTot[1];
				convG[(i*320)+j+2]=(UINT8)resTot[2];
				convG[(i*320)+j+3]=(UINT8)resTot[3];
				convG[(i*320)+j+4]=(UINT8)resTot[4];
				convG[(i*320)+j+5]=(UINT8)resTot[5];
				convG[(i*320)+j+6]=(UINT8)resTot[6];
				convG[(i*320)+j+7]=(UINT8)resTot[7];
				/////////////////////

				//Top row
				ptr_1=(__m256*)&fB[((i-1)*320)+j-1];
				ptr_2=(__m256*)&fB[((i-1)*320)+j];
				ptr_3=(__m256*)&fB[((i-1)*320)+j+1];
				resTR = _mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));
				

				//Midle row
				ptr_1=(__m256*)&fB[((i)*320)+j-1];
				ptr_2=(__m256*)&fB[((i)*320)+j];
				ptr_3=(__m256*)&fB[((i)*320)+j+1];
				resMR=_mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));

				//Midle row - Pixels of Interest
				resValue=_mm256_mul_ps(*ptr_2,ck);
				
				//Bottom row
				ptr_1=(__m256*)&fB[((i+1)*320)+j-1];
				ptr_2=(__m256*)&fB[((i+1)*320)+j];
				ptr_3=(__m256*)&fB[((i+1)*320)+j+1];
				resBR=_mm256_add_ps(*ptr_1, (_mm256_add_ps(*ptr_2, *ptr_3)));
				
				resTot=_mm256_add_ps(resBR, (_mm256_add_ps(resTR, resMR)));

				resTot=_mm256_mul_ps(resTot,kv);

				resTot=_mm256_add_ps(resTot,resValue);

				resTot = _mm256_max_ps(_min_mask, _mm256_min_ps(_max_mask, resTot));

				convB[(i*320)+j]=(UINT8)resTot[0];
				convB[(i*320)+j+1]=(UINT8)resTot[1];
				convB[(i*320)+j+2]=(UINT8)resTot[2];
				convB[(i*320)+j+3]=(UINT8)resTot[3];
				convB[(i*320)+j+4]=(UINT8)resTot[4];
				convB[(i*320)+j+5]=(UINT8)resTot[5];	
				convB[(i*320)+j+6]=(UINT8)resTot[6];	
				convB[(i*320)+j+7]=(UINT8)resTot[7];	
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

