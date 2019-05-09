
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv\highgui.h>
#include <opencv\cxcore.h>
#include <opencv\cv.h>
#include <opencv2\core\mat.hpp>

using namespace cv;
using namespace std;

#define clipping(x) x>255?255:x<0?0:x

#define WIDTH 1920 // 1920 or 416
#define HEIGHT 1080 // 1080 or 240
#define NUM_FRAME 240 //240 or 300

#define VIDEO_NUM 1

#define USEGPU 1

cudaError_t addWithCuda(unsigned char *outArr, unsigned char *inputArr, unsigned int size);


__global__ void addKernel(unsigned char *dev_out, unsigned char *dev_in, int VideoSize)
{
	//https://en.wikipedia.org/wiki/YUV#Y.E2.80.B2UV420p_.28and_Y.E2.80.B2V12_or_YV12.29_to_RGB888_conversion 참조!
	
	int pixel_ROW = blockIdx.x * blockDim.x + threadIdx.y;   //x 픽셀 좌표. 
	int pixel_COLUMN = blockIdx.y * blockDim.y + threadIdx.x; //y 픽셀 좌표.

	if (pixel_COLUMN >= HEIGHT || pixel_ROW >= WIDTH) return; //이게 없으면 코드가 더 돌아서 없는 영역을 0으로 바꾸고 해서 문제가 생김!
															// 해상도 외의 영역을 반드시 잘라 줄 필요가 있음!!!

	unsigned char y = dev_in[pixel_COLUMN*WIDTH + pixel_ROW];
	unsigned char u = dev_in[(int)(pixel_COLUMN / 2)*(WIDTH / 2) + (int)(pixel_ROW / 2) + (WIDTH*HEIGHT)];
	unsigned char v = dev_in[(int)(pixel_COLUMN / 2)*(WIDTH / 2) + (int)(pixel_ROW / 2) + (WIDTH*HEIGHT) + ((WIDTH*HEIGHT) / 4)];

	unsigned char b = y + (1.370705 * (v - 128));
	unsigned char g = y - (0.698001 * (v - 128)) - (0.337633 * (u - 128));
	unsigned char r = y + (1.732446 * (u - 128));
	dev_out[pixel_COLUMN*WIDTH + pixel_ROW] = b;
	dev_out[pixel_COLUMN*WIDTH + pixel_ROW + (WIDTH*HEIGHT)] = g;
	dev_out[pixel_COLUMN*WIDTH + pixel_ROW + (WIDTH*HEIGHT) + (WIDTH*HEIGHT)] = r;

}

int main()
{
	cudaError_t cudaStatus;
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

	//////////////////////////////////////////////////////////////////

	unsigned char **frame_no_loss_yuv;

	frame_no_loss_yuv = new unsigned char *[NUM_FRAME]; //해당 동영상 만큼의 프레임을 읽어오는 코드.

	for (int i = 0; i < NUM_FRAME; i++) {
		frame_no_loss_yuv[i] = new unsigned char[WIDTH*HEIGHT * 3 / 2];
	}

	// Kimono1_1920x1080_24.yuv or rec.yuv
	FILE* infile = fopen("Kimono1_1920x1080_24.yuv", "rb"); //YUV를 바이너리 파일로 읽음
	if (!infile) { //파일 없으면
		printf("There isn't file!\n");
	}
	for (int i = 0; i < NUM_FRAME; i++) {
		fread(frame_no_loss_yuv[i], 1, WIDTH*HEIGHT * 3 / 2, infile);
#if !USEGPU //CPU로 동작 할 때
		Mat mYUV(HEIGHT + HEIGHT / 2, WIDTH, CV_8UC1, (void*)frame_no_loss_yuv[i]);
		Mat mRGB(HEIGHT, WIDTH, CV_8UC3);
		cvtColor(mYUV, mRGB, CV_YUV2RGB_YV12, 3);

#else //GPU로 동작 할 때
		unsigned char *ArrRGB = new unsigned char[WIDTH * HEIGHT * 3 * 1];
		cudaStatus = addWithCuda(ArrRGB, frame_no_loss_yuv[i], WIDTH*HEIGHT); // 기본코드. 배열을 집어넣고 쿠다 연산을 했을때 out값에 잘 실행 됐는지의 여부가 나오는데 그 코드이다.
		if (cudaStatus != cudaSuccess) { //잘 안됐엉
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		// 이 상태에서는 ArrRGB 배열에는 B/G/R 통 프레임 순으로 담겨 있다. 

		

		//unsignedchar* to MATRIX 
		//ANSWER -> cv::Mat my_mat(rows, cols, CV_8UC1, &buf[0]); //in case of BGR image use CV_8UC3
		//cvtColor(frame, frame, CV_RGB2BGR);


		//http://stackoverflow.com/questions/15821253/merging-three-grayscale-r-g-b-images-into-a-single-color-image-in-opencv 참고
		Mat matB(HEIGHT, WIDTH, CV_8UC1, &ArrRGB[0]); // 
		Mat matG(HEIGHT, WIDTH, CV_8UC1, &ArrRGB[WIDTH*HEIGHT]); //
		Mat matR(HEIGHT, WIDTH, CV_8UC1, &ArrRGB[WIDTH*HEIGHT*2]); //

		vector<Mat> array_to_merge;

		array_to_merge.push_back(matR);
		array_to_merge.push_back(matG);
		array_to_merge.push_back(matB);

		Mat mRGB;
		merge(array_to_merge, mRGB);

		free(ArrRGB);

		matB.release();
		matG.release();
		matR.release();
#endif
	
		imshow("DISPLAY_YUV", mRGB); //temp
		cvWaitKey(1); // 이게 없으면 코드가 안돌음.
		
#if !USEGPU
		mYUV.release();
#endif
		mRGB.release(); //temp
		
	}
	//혹시 fseek 쓸거면 http://forum.falinux.com/zbxe/index.php?document_srl=408250&mid=C_LIB 참조


	///////////////////////////////////////////////////////////////////
    // Add vectors in parallel.
    

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
#if USEGPU

#endif

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(unsigned char *outArr, unsigned char *inputArr, unsigned int VideoSize)
{
	int maxThreadsX, maxThreadsY;
	cudaDeviceGetAttribute(&maxThreadsX, cudaDevAttrMaxGridDimX, 0);
	cudaDeviceGetAttribute(&maxThreadsY, cudaDevAttrMaxGridDimY, 0);

	//Set block size and grid size to size of the image. 
	int maxnThreads = 32; // 디폴트 32x32

	float XSize = ceil(WIDTH / 32.0f);
	float YSize = ceil(HEIGHT / 32.0f);

	if (XSize > maxThreadsX)
		XSize = maxThreadsX;
	if (YSize > maxThreadsY)
		YSize = maxThreadsY;

	dim3 nBlocks(XSize, YSize);
	//   nBlocks.x nBlocks.y 확인 
	dim3 nThreads(maxnThreads, maxnThreads); // 쓰레드 32x32 디폴트.


    unsigned char *dev_in = 0; // input YUV
    unsigned char *dev_out = 0; // output RGB
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_in, VideoSize * sizeof(unsigned char) * 3 / 2); //YUV input
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, VideoSize * sizeof(unsigned char) * 3 * 1); //RGB output Malloc
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    //cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMalloc failed!");
    //    goto Error;
    //} 
	// USELESS CODE!!

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, inputArr, VideoSize * sizeof(unsigned char) * 3 / 2, cudaMemcpyHostToDevice); //yuv gpu memory copy
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	
    //cudaStatus = cudaMemcpy(dev_out, outArr, VideoSize * sizeof(unsigned char) * 3 * 1, cudaMemcpyHostToDevice); //rgb gpu memory copy
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}
	// USELESS CODE!!


    // Launch a kernel on the GPU with one thread for each element.
	

    addKernel<<< nBlocks, nThreads >>>(dev_out, dev_in, VideoSize); //BlockSize 수 & Thread Size 수 이건 사람마다 해석이 달라서 내 해석대로 함!

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outArr, dev_out, VideoSize * sizeof(unsigned char) * 3 * 1, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_in);
    cudaFree(dev_out);
    

    return cudaStatus;
}
