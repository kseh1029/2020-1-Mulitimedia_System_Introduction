# HW 4 : DCT lossy compression

#### 과제에 대한 설명

◦ 수업에서 배운 DCT변환을 이용하여 영상의 lossy compressio을 구현

◦ 영상 입력, 출력, 처리 등은 OpenCV 사용

◦ 구현 순서

1) 2D DCT에 사용될 8x8 DCT basis를 구한다. (전체 64개)

2) 입력영상을 read 하여 graylevel로 변환한다.

3) 영상의 8x8 영역(block)에 대하여 f(i,j)를 8x8 DCT basis 를 이용하여

8x8 주파수 공간 F(u,v)로 변환한다.

4) F(u,v)를 다시 IDCT로 8x8 영상을 다시 복원한다.

- 이때, F(u,v)의 8x8, 4x4, 2x2 값들만 사용하여 비교

5) 영상 전체에 대하여 3)-4) 과정을 반복한다.

6) 입력 1장 영상에 대하여 복원된 영상 3장을 각각 저장한다.

![HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture11.jpeg](/Users/seunghyun/Downloads/Export-9adbe113-cb90-4753-b3c2-a5330e6c66ed/Import Nov 3, 2020 5be5c2bb0a854328bb642adcf106711f/HW4_How 6b1abd62aa6445e8bc7c312349c3718e/picture11.jpeg)

<img src="HW4_How 6b1abd62aa6445e8bc7c312349c3718e/picture12.jpeg" alt="HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture12.jpeg" style="zoom:80%;" />



![HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture15.jpeg](HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture15.jpeg)

![HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture18.jpeg](HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture18.jpeg)

![HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture21.jpeg](HW4_How%206b1abd62aa6445e8bc7c312349c3718e/picture21.jpeg)

제출 할 것

1) Test 영상은 2장 (Lena 및 직접 찍은 영상)

- 영상 크기는 x기준으로 1000이하

2) 각 test 영상마다 3개의 f(i,j) 복원 결과

3) 8x8 DCT basis는 v=0일때 u를 0~7까지 8개의 matrix를 보고서에 출력

4) 그리고 source code 출력



Ⅰ. 과제 결과

**Grayscaled Image**

![HW4%20c54a81b00d9549308de3ec4008bf082b/image77.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image77.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image78.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image78.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image79.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image79.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image80.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image80.bmp)

**DCT에서 8 X 8 가 유효할 때**

**DCT에서 4 X 4 가 유효할 때**

**DCT에서 2 X 2 가 유효할 때**

**DCT에서 2 X 2 가 유효할 때**

![HW4%20c54a81b00d9549308de3ec4008bf082b/image81.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image81.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image82.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image82.bmp)

**DCT에서 4 X 4 가 유효할 때**

**DCT에서 8 X 8 가 유효할 때**

![HW4%20c54a81b00d9549308de3ec4008bf082b/image83.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image83.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image84.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image84.bmp)

**Grayscaled Image**

Ⅱ. HW #5 목적, 접근방법 및 세부결과

해당 HW #5 의 DCT 는 Pixel을 Frequency Domain 으로 변환해 고주파 부분을 제거하여 사진의 용량을 압축하는 방법이다. 이를 다시 IDCT 하여 Pixel 로 변환해주면 이미지로 출력이 가능하다.

고주파를 없애면서 이미지를 압축하는 방식이므로 lossy compression 에 해당된다.

> 다음의 절차를 통해 압축 결과를 얻을 수 있다.1) 입력 이미지를 Grayscale 로 변환한다.2) Grayscale 이미지를 DCT 한다.3) DCT 된 이미지에서 필요한 부분을 제외하고 0으로 바꾼다.4) IDCT 하여 영상을 복원한다.2D Discrete Cosine Transform (2D DCT) 공식$F(u,v) = \frac{C(u)C(v)}{4}\sum_{i = 0}^{7}{\sum_{j = 0}^{7}\cos}\frac{(2i + 1)u\pi}{16}\cos\frac{(2j + 1)v\pi}{16}f(i,j)$2D Inverse Discrete Cosine Transform (2D IDCT) 공식$\widetilde{f}(i,j) = \sum_{u = 0}^{7}{\sum_{v = 0}^{7}\frac{C(u)C(v)}{4}}\cos\frac{(2i + 1)u\pi}{16}\cos\frac{(2j + 1)v\pi}{16}F(u,v)$$C(x) = \frac{1}{\sqrt{2}}\,(x = 0),\,\text{otherwise}\, 1$과제 진행에 앞서 DCT를 만들어서 제대로 작동하는지 확인이 필요했다.위와 같은 코드를 통해 DCT 작동 여부를 확인하였다.DCT 결과에서 나누기를 1000 하여 오른쪽 dctblock=1.0e+003* 이미지와 모양을 맞춘 모습이다.정상적으로 DCT 가 진행되었음을 확인할 수 있다.이렇게 설계된 DCT를 이용해 실제 이미지에 적용을 해봤다.lena를 불러와서 f(0,0) ~ f(7,7) 까지 DCT, IDCT를 적용한 것의 각 픽셀의 값을 확인한 결과이다.※ 편의 상 원본이미지가 Pixel[y][x] 였을 때, 바로 DCT[u][[v] 로 변환하고 IDCT[i,j] 로 복원하였다즉 f(y,x) => F(u,v) => f(i,j) 가 된 것 이므로 y축으로 커질수록 u 가 증가하는 것이다.

![HW4%20c54a81b00d9549308de3ec4008bf082b/image85.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image85.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image86.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image86.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image87.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image87.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image88.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image88.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image89.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image89.bmp)

**1) lena.jpg를 DCT에서 8 X 8 살릴 때**

> DCT에서 수정을 하지않고 IDCT를 한 결과 원본이미지와 같은 결과를 나타내었다.F(u,v)에서 F(0,0) ~ F(7,0)을 구하기 위해서는 세로방향으로 8개의 값을 확인하면 된다.

![HW4%20c54a81b00d9549308de3ec4008bf082b/image90.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image90.bmp)

![HW4%20c54a81b00d9549308de3ec4008bf082b/image91.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image91.bmp)

**2) lena.jpg를 DCT에서 4 X 4 살릴 때**

> 

![HW4%20c54a81b00d9549308de3ec4008bf082b/image92.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image92.bmp)

DCT에서 4x4 배열을 제외한 나머지 고주파 부분을 0으로 변환한 후 IDCT 하였다.

IDCT를 정수로 바꾼 8 by 8 픽셀을 확인해보면 8bit 원본이미지의 8 by 8 픽셀과 큰 차이가 없다.

이미지 용량은 $\frac{4*4}{8*8} = \frac{16}{64}$ 배로 감소하였지만 이미지를 못알아 볼 정도로 손상이 되지는 않았다.

이는 DC 값 (1284.749878) 이 이미지의 전반적인 모양을 담당하고 있기 때문이다.

**3) lena.jpg를 DCT에서 2 X 2 살릴 때**

![HW4%20c54a81b00d9549308de3ec4008bf082b/image93.bmp](HW4%20c54a81b00d9549308de3ec4008bf082b/image93.bmp)

DCT에서 2x2 배열을 제외한 나머지 고주파 부분을 0으로 변환한 후 IDCT 하였다.

이미지 용량은 $\frac{2*2}{8*8} = \frac{4}{64}$ 배로 감소하였고 4x4 배열에 비해 손실도가 더 크다.

Ⅲ. C++ 코드

/* 2017117876 김승현 HW#4

2D DCT & 2D IDCT */

#include <opencv2/opencv.hpp>

#include <iostream>

#define _USE_MATH_DEFINES

#include <math.h>

#include <cmath>

using namespace cv;

int main(void) {

Mat input = imread("lena.jpg", IMREAD_COLOR);

Mat conv_to_gray;

if (input.data == NULL) {

std::cout << "해당 사진이 없습니다." << std::endl;

return -1;

}

cvtColor(input, conv_to_gray, COLOR_BGR2GRAY);

namedWindow("Input Image");

imshow("Input Image", input);

namedWindow("grayscale");

imshow("grayscale", conv_to_gray);

int i, j, u, v;

int test = 0; // N*test X N*test 의 픽셀을 도스창에 보여준다.

int live = 4; // DCT 에서 live 배열만큼만 살린다

int N = 8; // Size

int height = conv_to_gray.rows;

int width = conv_to_gray.cols;

int x_axis_number = width / N;

int y_axis_number = height / N;

int x_count = 0, y_count = 0;

float Coeffcient[8] = { (1 / (sqrt(2))),1,1,1,1,1,1,1 };

float temp;

float** Pixel = (float**)malloc(sizeof(float*) * height);

for (int k = 0; k < height; k++)

{

Pixel[k] = (float*)malloc(sizeof(float) * width);

}

float** DCT = (float**)malloc(sizeof(float*) * height);

for (int k = 0; k < height; k++)

{

DCT[k] = (float*)malloc(sizeof(float) * width);

}

float** IDCT = (float**)malloc(sizeof(float*) * height);

for (int k = 0; k < height; k++)

{

IDCT[k] = (float*)malloc(sizeof(float) * width);

}

// Pixel 에 원본이미지 복사

for (int y = 0; y < height; y++) {

for (int x = 0; x < width; x++) {

Pixel[y] = conv_to_gray.at<uchar>(y, x);

}

}

// 원본 이미지 grayscale 배열 출력

printf("8bit 원본이미지의 %d by %d 픽셀", test + N, test + N);

for (i = 0; i < N; i++) {

printf("\n");

for (j = 0; j < N; j++) {

printf("%f ", Pixel[test * N + i][test * N + j]);

}

}

// 2D Discrete Cosine Transform

for (y_count = 0; y_count < y_axis_number; y_count++) {

for (x_count = 0; x_count < x_axis_number; x_count++) {

for (u = 0; u < N; u++) {

for (v = 0; v < N; v++) {

temp = 0.0;

for (i = 0; i < N; i++) {

for (j = 0; j < N; j++) {

temp += cos((2 * i + 1) * (u * M_PI) / (2 * N)) * cos((2 * j + 1) * (v * M_PI) / (2 * N)) * Pixel[y_count * N + i][x_count * N + j];

}

}

temp *= (Coeffcient[u] * Coeffcient[v]) / 4;

DCT[y_count * N + u][x_count * N + v] = temp;

}

}

}

}

// live 배열 만큼만 살린다.

for (y_count = 0; y_count < y_axis_number; y_count++) {

for (x_count = 0; x_count < x_axis_number; x_count++) {

for (u = 0; u < N; u++) {

for (v = 0; v < N; v++) {

if (u >= live || v >= live)

DCT[y_count * N + u][x_count * N + v] = 0;

}

}

}

}

// DCT 배열 출력

printf("\n\nDCT 를 적용한 %d by %d 픽셀", test + N, test + N);

for (u = 0; u < N; u++) {

printf("\n");

for (v = 0; v < N; v++) {

printf("%f ", DCT[test * N + u][test * N + v]);

}

}

// 2D Inverse Discrete Cosine Transform

for (y_count = 0; y_count < y_axis_number; y_count++) {

for (x_count = 0; x_count < x_axis_number; x_count++) {

for (i = 0; i < N; i++) {

for (j = 0; j < N; j++) {

temp = 0.0;

for (u = 0; u < N; u++) {

for (v = 0; v < N; v++) {

temp += ((Coeffcient[u] * Coeffcient[v]) / 4) * cos((2 * i + 1) * (u * M_PI) / (2 * N)) *

cos((2 * j + 1) * (v * M_PI) / (2 * N)) * DCT[y_count * N + u][x_count * N + v];

}

}

IDCT[y_count * N + i][x_count * N + j] = temp;

}

}

}

}

// IDCT 배열 출력

printf("\n\nIDCT 를 적용한 %d by %d 픽셀", test + N, test + N);

for (i = 0; i < N; i++) {

printf("\n");

for (j = 0; j < N; j++) {

printf("%f ", IDCT[test * N + i][test * N + j]);

}

}

// IDCT 배열 정수화

Mat result(height, width, CV_8U);

for (i = 0; i < height; i++) {

for (j = 0; j < width; j++) {

result.at<uchar>(i, j) = char(round(IDCT[i][j]));

}

}

// 정수화 된 IDCT 배열 출력

printf("\n\nIDCT 를 정수로 바꾼 %d by %d 픽셀", test + N, test + N);

for (i = 0; i < N; i++) {

printf("\n");

for (j = 0; j < N; j++) {

printf("%d ", result.at<uchar>(test * N + i, test * N + j));

}

}

namedWindow("result");

imshow("result", result);

waitKey(0);

return 0;

}