# HW 2 : Error Diffusion Dithering

#### 과제에 대한 설명

구현방법

◦ 수업에서 다루었던 오차확산 디더링을 C/C++/java를 이용하여

구현

◦ 영상 입력, 출력, 픽셀에 대한 연산 처리는 OpenCV 사용

◦ 오차확산 디더링 알고리즘은 직접 구현

◦ 입력영상 2개: lena.jpg 및 직접 찍은 영상 1장 (가로 해상도 1000

픽셀 이하)

1 ) 컬러영상을 read 후 흑백(grey level)영상으로 변환

2) 디더링 알고리즘을 적용

3) 디더링 결과는 Greylevel 영상으로 저장

4) 결과영상을 imageviewer로 확인 후 HWP로 복사

추가점수

- 컬러영상을 greylevel로 변환하지 말고 각 RGB 채널

을 greylevel 영상으로 간주하여 각 채널을 디더링

- 3개의 디더링 결과를 다시 RGB 영상으로 합성하여

color dithering 구현

- RGB를 분리하고 합성하는 것은 OpenCV 사용

**lena 원본**

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image63.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image63.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image64.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image64.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image65.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image65.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image66.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image66.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image67.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image67.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image68.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image68.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image69.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image69.bmp)

과제 결과

**lena Grayscale**

**lena 1 bit dithering**

**lena Red Single Channel**

**lena Green Single Channel**

**lena Blue Single Channel**

**lena 3 bit dithering**

**sedra 원본**

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image70.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image70.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image71.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image71.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image72.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image72.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image73.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image73.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image74.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image74.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image75.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image75.bmp)

![HW2%206c9b8a5f687848d08c2f54b06db75fc2/image76.bmp](HW2%206c9b8a5f687848d08c2f54b06db75fc2/image76.bmp)

**sedra Grayscale**

**sedra 3 bit dithering**

**sedra 1 bit dithering**

**sedra Red Single Channel**

**sedra Green Single Channel**

**sedra Blue Single Channel**

해당 HW #2 의 목표인 24bit Full Color 이미지를 grayscale 로 변환하여 Error Diffusion dithering을 구현하였다. 기존에 제공된 lena.jpg 와 직접 촬영한 sedra.jpg 를 이용하여 테스트 한 결과이다.

추가적으로, split 함수를 이용하여 R,G,B를 Single Channel 으로 분리시켜 Error Diffusion Dithering을 진행한다음 다시 merge 함수를 사용하여 3 bit dithering 도 구현하였다.

Ⅱ. HW#2 목적, 접근방법 및 세부결과

이 과제의 목적은 Error Diffusion Dithering 이론을 실제로 OpenCV를 통해 주어진 사진과 직접 찍은 사진을 C/C++/Java를 통해 적용해 보는 것이다. (가로 해상도 1000 픽셀 이하)

과제를 위해 C++를 주요 사용언어로 선택하였으며, Windows 10 환경에서 Visual Studio 2015 버전에 OpenCV 4.3.0 버전을 설치하여 진행하였다.

**① OpenCV 기초와 명령어**

처음 사용하는 프로그램인 만큼 OpenCV 의 기초에 대해 학습할 필요가 있었다.

프로젝트에서 릴리즈와 디버그의 환경변수를 수정하고, 사용자 라이브러리를 추가하여 OpenCV를 사용할 수 있게 하였다.

사용에 앞서, 간단한 라이브러리에 대해 알아봤다.

#include를 할 때 opencv, opencv2 두가지가 있었는데 opencv 는 C 언어에 사용하고, opencv2 는 C++에 사용한다는 차이점이 있다.

#include <opencv2/core.hpp> 는 이미지 저장시 사용되는 Class를 비롯해 OpenCV에서 사용하는 기본적인 자료구조와 함수들이 포함되어 있는 라이브러리이다.

#include <opencv2/imgcodes.hpp> 는 이미지 파일을 읽어오는 함수, 이미지 파일로 저장하는 함수를 포함하는 라이브러리이다.

#include <opencv2/highgui.hpp> 는 인터페이스를 구현해둔 모듈로서 화면에 이미지를 출력하는 라이브러리이다.

이 세 가지를 모두 포함하고 있는 라이브러리는 #include <opencv2/opencv.hpp> 이므로 이 라이브러리만 사용해도 이 과제를 해결하는데 큰 문제가 없다.

OpenCV에서 이미지를 불러와서 보관하는 배열의 자료형은 Mat 이다.

Mat.rows() 로 해당 이미지 픽셀의 열을 뱉어낼 수 있고, Mat.columns() 로 해당 이미지 픽셀의 행을 뱉어낼 수 있다. 그리고, Mat.channels()를 통해 채널 수를 구할 수 있다. 일반적으로 Grayscale 은 Single 채널이며, RGB 의 경우 3 채널, 여기에 알파 채널이 포함되면 4 채널이 된다. 알파 채널은 밝기를 나타낸다.

Mat 에 이미지를 불러오기 위해서는 imread 함수를 사용해야 한다.

cv::imread (“파일명”, image type); 으로 사용할 수 있다.

여기서 자주쓰는 image type 는 3가지로 분류할 수 있다.

IMREAD_UNCHANGED : 알파채널을 포함한 컬러 이미지로 불러온다.

IMREAD_COLOR : 알파채널을 제외한 컬러 이미지로 불러온다.

IMREAD_GRAYSCALE : 그레이 스케일로 변환한 이미지를 불러온다.

이 과제에서 lena.jpg를 읽어야 하므로 Mat input; 이라는 배열을 만들었다고 치면,

input = (“lena.jpg”, IMREAD_COLOR); 로 lena.jpg를 24bit 컬러 이미지로 불러올 수 있다.

Mat 로 선언한 배열명에 empty() 함수를 사용하면 빈 배열인지 아닌지 확인할 수 있다.

따라서, if 문과 input.empty()를 조합하면 빈 이미지 파일을 불러오는 것을 방지할 수 있다.

사진을 저장하지 않고 Window 로 결과를 나타나게 할 수 있다.

cv::namedWindow(“Title”, WINDOW_AUTOSIZE);

cv::imshow(“Title”, 배열명);

cv::waitKey(0);

여기서 namedWindow 는 Title 이라는 제목의 빈 창을 띄운다.

imshow를 통해 namedWindow 에 의해 띄어진 Title 이라는 제목의 빈 창에 배열명에 해당하는 이미지를 나타나게 한다.

waitkey는 그 창의 대기시간이며, 단위는 ms 이다. 0을 적을땐 닫을 때 까지 유지된다.

WINDOW_AUTOSIZE 대신에 WINDOW_NORMAL을 사용하면 사이즈 조절이 가능하다.

최종적으로 배열에 있는 이미지를 저장하기 위해서는 cv::imwrite(“파일명”, 배열명); 으로 저장할 수 있다.

이미지를 변환하기 위해서는 cvtColor를 사용할 수 있다.

cv::cvtColor(대상배열명,저장할배열명,변환방식); 으로 사용할 수 있으며

RGB에서 Grayscale 로 바꾸기 위해서는 CV_BGR2GRAY를 사용한다.

배열명.at(row,col)[channel] 을 통해 해당 픽셀의 값을 얻어내거나 수정할 수 있다.

split(배열명A, 배열명B); 을 통해 배열명A에 있던 채널을 배열명B[0], 배열명B[1], 배열명B[2] 로 각각 B, G, R 로 채널을 분리시킬 수 있다.

merge(대상배열, 채널수, 출력배열); 로 대상배열의 채널수 만큼 합쳐서 출력배열로 내보낸다.

그리고, C++ 의 기본 iostream 과 OpenCV 의 클래스가 겹치는 이름이 없어서 using namespace cv; 를 선언해두어 편리하게 사용해도 문제가 없다.

**② Error Diffusion Dithering 이론**

[Untitled](HW2%206c9b8a5f687848d08c2f54b06db75fc2/Untitled%20Database%2091e302db1f2044389253704c1fb5f2e5.csv)

왼쪽과 같은 가충지창 6칸이 한 세트가 되어 이미지 전체를 한칸 한칸 탐험한다고 생각하면 된다.

8bit grayscale 기준으로 기존 원본 이미지에서 각각이 가진 픽셀이 g(x,y) (0~255 인 값) 이다.

픽셀 g(a,b) 의 값이 127 보다 크다면 g(a,b)-255 가 오차 e이며

그 외에는 g(a,b) 가 오차 e가 된다.

이 오차 e(a,b)에 가중치 $\frac{7}{16}$, $\frac{3}{16}$, $\frac{5}{16}$, $\frac{1}{16}$ 를 각각 에 곱한다음 각각 g(a+1,b), g(a-1,b+1), g(a,b+1), g(a+1,b+1) 에 다시 더해준다.

이런식으로 모든 이미지를 처리해준다.

그리고 나서 각각이 가진 픽셀을 비교했을 때 127 이하일 경우 0, 127보다 클 경우 1로 binary 하게 변환해주면 된다.

한가지 예를 들어 다음과 같은 5x5 이미지가 있다고 가정하면 6칸 까지 가중치 창이 p에서부터 시작하기 위해선

[Untitled](HW2%206c9b8a5f687848d08c2f54b06db75fc2/Untitled%20Database%2097e1160db5fc464caf9139bfa79e8328.csv)

p 가 최소한 g(1,0)에서부터 시작해야하고,

그 위치에서

x 축으로 끝까지 갔을땐 p가 g(3,0)

y 축으로 끝까지 갔을땐 p가 g(1,3) 가 한계임을 알 수 있다.

이를 통해 나중에 Error Diffusion Dithering을 하면서 5x5 행렬의 경우

x 축은 1 ~ 3까지,

y 축은 0 ~ 3 까지 가중치를 계산해야 한다.

즉, x=1 ~ n-2 까지, y=0 ~ n-2 까지 반복하는 것이다.

이를 for 문으로 나타내면 y = 0; y < height - 1; y++, x = 1; x < width - 1; x++ 로 만들 수 있을 것이다.

그렇게 for 문을 돌면서 if (arr[y] > 127) 면 e = arr[y] - 255; 그렇지 않으면 e = arr[y]; 가 되게 하고

arr[y][x + 1] += e * 7 / 16;

arr[y + 1][x - 1] += e * 3 / 16;

arr[y + 1] += e * 5 / 16;

arr[y + 1][x + 1] += e / 16;

위와 같이 각 픽셀에 오차를 더해주면 될 것이다.

그리고나서, 모든 픽셀에 오차가 고려가 됐으면 if ((arr[y]) > 127) 면 255, 아니면 0 으로 binary 하게 만들어 주면 된다.

Ⅲ. C++ 코드

/* 2017117876 김승현 HW#2

Error Diffusion Dithering */

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

void Error_Diffusion_Dithering(Mat &image) {

int y, x;

int height = image.rows;

int width = image.cols;

double e;

double **arr = (double**)malloc(sizeof(double*) * height);

for (int k = 0; k < height; k++)

{

arr[k] = (double*)malloc(sizeof(double) * width);

}

for (y = 0; y < height - 1; y++) {

for (x = 1; x < width - 1; x++) {

arr[y] = image.at<uchar>(y, x);

}

}

for (y = 0; y < height - 1; y++) {

for (x = 1; x < width - 1; x++) {

if (arr[y] > 127)

e = arr[y] - 255;

else

e = arr[y];

arr[y][x + 1] += e * 7 / 16;

arr[y + 1][x - 1] += e * 3 / 16;

arr[y + 1] += e * 5 / 16;

arr[y + 1][x + 1] += e / 16;

}

}

for (y = 0; y < height; y++) {

for (x = 0; x < width; x++) {

if ((arr[y]) > 127)

image.at<uchar>(y, x) = 255;

else

image.at<uchar>(y, x) = 0;

}

}

for (int k = 0; k<y; k++) {

free(arr[k]);

}

free(arr);

}

int main(void) {

Mat input = imread("lena.jpg", IMREAD_COLOR);

Mat conv_to_gray;

if (input.data == NULL) {

std::cout << "해당 사진이 없습니다." << std::endl;

return -1;

}

if (input.cols > 1000) {

std::cout << "가로 해상도 1000픽셀 이하 이미지를 이용하세요" << std::endl;

return -1;

}

cvtColor(input, conv_to_gray, COLOR_BGR2GRAY);

namedWindow("Original");

imshow("Original", input);

namedWindow("Grayscale");

imshow("Grayscale", conv_to_gray);

Mat bgr[3];

split(input, bgr);

Error_Diffusion_Dithering(conv_to_gray);

Error_Diffusion_Dithering(bgr[2]);

Error_Diffusion_Dithering(bgr[1]);

Error_Diffusion_Dithering(bgr[0]);

namedWindow("Red Single Channel");

namedWindow("Green Single Channel");

namedWindow("Blue Single Channel");

imshow("Red Single Channel", bgr[2]);

imshow("Green Single Channel", bgr[1]);

imshow("Blue Single Channel", bgr[0]);

Mat total[] = { bgr[0], bgr[1], bgr[2] };

Mat output;

merge(total, 3, output);

namedWindow("RGB_Result");

imshow("RGB_Result", output);

namedWindow("Grayscale_Result");

imshow("Grayscale_Result", conv_to_gray);

waitKey(0);

imwrite("gray_result.jpg", conv_to_gray);

imwrite("RGB_result.jpg", output);

return 0;

}