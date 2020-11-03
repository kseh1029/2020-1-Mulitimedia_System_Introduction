# 2020년도 1학기 : 멀티미디어시스템개론 (EECS462001)

* ## [과제 솔루션 보기](HW/README.md)

* ## Repository 에 대한 설명

본 자료는 경북대학교 전자공학부 수업 중 2020년도 1학기에 진행한 ‘박순용’ 교수님의 ‘멀티미디어시스템개론’ 과목에 대한 정리이다.

상단의 폴더를 클릭하며 과제에 대한 설명과 해결 과정을 확인할 수 있다.

아래의 표는 이 과목이 어떤 식으로 진행이 되었는지에 대한 강의계획서이다. 

![01](./images/01.png )

![02](./images/02.png )

# Ⅰ. 과제 결과

![https://blog.kakaocdn.net/dn/NI4aR/btqLA25tGGs/cauVfMpKPIGEHD1cBOAt71/img.png](https://blog.kakaocdn.net/dn/NI4aR/btqLA25tGGs/cauVfMpKPIGEHD1cBOAt71/img.png)

![https://blog.kakaocdn.net/dn/daXzYE/btqLveUx9jZ/S9YyN0kDcoWkrzOkMy9chk/img.png](https://blog.kakaocdn.net/dn/daXzYE/btqLveUx9jZ/S9YyN0kDcoWkrzOkMy9chk/img.png)

# Ⅱ. HW #5 목적, 접근방법 및 세부결과

해당 HW #5 의 DCT 는 Pixel을 Frequency Domain 으로 변환해 고주파 부분을 제거하여 사진의 용량을 압축하는 방법이다. 이를 다시IDCT 하여 Pixel 로 변환해주면 이미지로 출력이 가능하다.

고주파를 없애면서 이미지를 압축하는 방식이므로 lossy compression 에 해당된다.

다음의 절차를 통해 압축 결과를 얻을 수 있다.

1) 입력 이미지를 Grayscale 로 변환한다.

2) Grayscale 이미지를 DCT 한다.

3) DCT 된 이미지에서 필요한 부분을 제외하고 0으로 바꾼다.

4) IDCT 하여 영상을 복원한다.

2D Discrete Cosine Transform (2D DCT) 공식

![https://blog.kakaocdn.net/dn/bu8mD0/btqLzGB3Xpe/P7f9dqMn5erfc6v7w5OVOK/img.jpg](https://blog.kakaocdn.net/dn/bu8mD0/btqLzGB3Xpe/P7f9dqMn5erfc6v7w5OVOK/img.jpg)

2D Inverse Discrete Cosine Transform (2D IDCT) 공식

![https://blog.kakaocdn.net/dn/77MK7/btqLA3J5v7i/dKGXWVfpW7eCJe3sYRXMe0/img.jpg](https://blog.kakaocdn.net/dn/77MK7/btqLA3J5v7i/dKGXWVfpW7eCJe3sYRXMe0/img.jpg)

![https://blog.kakaocdn.net/dn/cJ3738/btqLxuoBZyE/JtNoX252SzodzHRk1IIXm0/img.jpg](https://blog.kakaocdn.net/dn/cJ3738/btqLxuoBZyE/JtNoX252SzodzHRk1IIXm0/img.jpg)

단,

![https://blog.kakaocdn.net/dn/bQCNfr/btqLvFj4bwR/TYAKk7jtN34yXLo6dumUkk/img.jpg](https://blog.kakaocdn.net/dn/bQCNfr/btqLvFj4bwR/TYAKk7jtN34yXLo6dumUkk/img.jpg)

과제 진행에 앞서 DCT를 만들어서 제대로 작동하는지 확인이 필요했다.

![https://blog.kakaocdn.net/dn/v3bZr/btqLA3ceyjz/Nv9Kjfdkv2mzzOEnKHym7k/img.jpg](https://blog.kakaocdn.net/dn/v3bZr/btqLA3ceyjz/Nv9Kjfdkv2mzzOEnKHym7k/img.jpg)

![https://blog.kakaocdn.net/dn/bnbyow/btqLA2RV846/AKDqZE3Z8V7SeY2o9lcVKk/img.jpg](https://blog.kakaocdn.net/dn/bnbyow/btqLA2RV846/AKDqZE3Z8V7SeY2o9lcVKk/img.jpg)

위와 같은 코드를 통해 DCT 작동 여부를 확인하였다.

![https://blog.kakaocdn.net/dn/b3pkSO/btqLzYvKHf6/6xIlhaR1FQVLy3PMBBKhFk/img.jpg](https://blog.kakaocdn.net/dn/b3pkSO/btqLzYvKHf6/6xIlhaR1FQVLy3PMBBKhFk/img.jpg)

DCT 결과에서 나누기를 1000 하여 오른쪽 dctblock=1.0e+003* 이미지와 모양을 맞춘 모습이다.

정상적으로 DCT 가 진행되었음을 확인할 수 있다.

이렇게 설계된 DCT를 이용해 실제 이미지에 적용을 해봤다.

lena를 불러와서 f(0,0) ~ f(7,7) 까지 DCT, IDCT를 적용한 것의 각 픽셀의 값을 확인한 결과이다.

![https://blog.kakaocdn.net/dn/kyonW/btqLwfrVb6e/qt88etUTe4NsjHZnm7CD60/img.jpg](https://blog.kakaocdn.net/dn/kyonW/btqLwfrVb6e/qt88etUTe4NsjHZnm7CD60/img.jpg)

※ 편의 상 원본이미지가 Pixel[y] 였을 때, 바로 DCT[u][[v] 로 변환하고 IDCT[i,j] 로 복원하였다

즉 f(y,x) => F(u,v) => f(i,j) 가 된 것 이므로 y축으로 커질수록 u 가 증가하는 것이다.

**1) lena.jpg를 DCT에서 8 X 8 살릴 때**

![https://blog.kakaocdn.net/dn/dknmHX/btqLzFJUKWh/D9eqWHNhoxrYoqaUgKlVv1/img.jpg](https://blog.kakaocdn.net/dn/dknmHX/btqLzFJUKWh/D9eqWHNhoxrYoqaUgKlVv1/img.jpg)

![https://blog.kakaocdn.net/dn/A6q7f/btqLwdVaFwX/G16aO0lK6npu0VnSgkCPX0/img.jpg](https://blog.kakaocdn.net/dn/A6q7f/btqLwdVaFwX/G16aO0lK6npu0VnSgkCPX0/img.jpg)

DCT에서 수정을 하지않고 IDCT를 한 결과 원본이미지와 같은 결과를 나타내었다.

F(u,v)에서 F(0,0) ~ F(7,0)을 구하기 위해서는 세로방향으로 8개의 값을 확인하면 된다.

**2) lena.jpg를 DCT에서 4 X 4 살릴 때**

![https://blog.kakaocdn.net/dn/NVPFb/btqLwdARZN5/TU3pUkPS4bYoW22cDcqik1/img.jpg](https://blog.kakaocdn.net/dn/NVPFb/btqLwdARZN5/TU3pUkPS4bYoW22cDcqik1/img.jpg)

DCT에서 4x4 배열을 제외한 나머지 고주파 부분을 0으로 변환한 후 IDCT 하였다.

IDCT를 정수로 바꾼 8 by 8 픽셀을 확인해보면 8bit 원본이미지의 8 by 8 픽셀과 큰 차이가 없다.

이미지 용량은

![https://blog.kakaocdn.net/dn/Gec04/btqLweGBY87/6jt2hUuEnMQh2Zx1wkRDrk/img.jpg](https://blog.kakaocdn.net/dn/Gec04/btqLweGBY87/6jt2hUuEnMQh2Zx1wkRDrk/img.jpg)

배로 감소하였지만 이미지를 못알아 볼 정도로 손상이 되지는 않았다.

이는 DC 값 (1284.749878) 이 이미지의 전반적인 모양을 담당하고 있기 때문이다.

**3) lena.jpg를 DCT에서 2 X 2 살릴 때**

![https://blog.kakaocdn.net/dn/ce9FkG/btqLyMP9bXr/DnmyR8IRvUMncSQwl4yq9K/img.jpg](https://blog.kakaocdn.net/dn/ce9FkG/btqLyMP9bXr/DnmyR8IRvUMncSQwl4yq9K/img.jpg)

DCT에서 2x2 배열을 제외한 나머지 고주파 부분을 0으로 변환한 후 IDCT 하였다.

이미지 용량은

![https://blog.kakaocdn.net/dn/4V41h/btqLvGced0i/081bauYBmwlJfZ18biInvK/img.jpg](https://blog.kakaocdn.net/dn/4V41h/btqLvGced0i/081bauYBmwlJfZ18biInvK/img.jpg)

배로 감소하였고 4x4 배열에 비해 손실도가 더 크다.

# Ⅲ. C++ 코드

/* 2017117876 김승현 HW#5

2D DCT & 2D IDCT */

#include <opencv2/opencv.hpp>

#include <iostream>

#define _USE_MATH_DEFINES

#include <math.h>

#include <cmath>

using namespace cv;

int main(void) {

Mat input = imread("lena.jpg", IMREAD_COLOR);

Mat conv_to_gray;

if (input.data == NULL) {

std::cout << "해당 사진이 없습니다." << std::endl;

return -1;

}

cvtColor(input, conv_to_gray, COLOR_BGR2GRAY);

namedWindow("Input Image");

imshow("Input Image", input);

namedWindow("grayscale");

imshow("grayscale", conv_to_gray);

int i, j, u, v;

int test = 0; // N*test X N*test 의 픽셀을 도스창에 보여준다.

int live = 4; // DCT 에서 live 배열만큼만 살린다

int N = 8; // Size

int height = conv_to_gray.rows;

int width = conv_to_gray.cols;

int x_axis_number = width / N;

int y_axis_number = height / N;

int x_count = 0, y_count = 0;

float Coeffcient[8] = { (1 / (sqrt(2))),1,1,1,1,1,1,1 };

float temp;

float** Pixel = (float**)malloc(sizeof(float*) * height);

for (int k = 0; k < height; k++)

{

Pixel[k] = (float*)malloc(sizeof(float) * width);

}

float** DCT = (float**)malloc(sizeof(float*) * height);

for (int k = 0; k < height; k++)

{

DCT[k] = (float*)malloc(sizeof(float) * width);

}

float** IDCT = (float**)malloc(sizeof(float*) * height);

for (int k = 0; k < height; k++)

{

IDCT[k] = (float*)malloc(sizeof(float) * width);

}

// Pixel 에 원본이미지 복사

for (int y = 0; y < height; y++) {

for (int x = 0; x < width; x++) {

Pixel[y] = conv_to_gray.at<uchar>(y, x);

}

}

// 원본 이미지 grayscale 배열 출력

printf("8bit 원본이미지의 %d by %d 픽셀", test + N, test + N);

for (i = 0; i < N; i++) {

printf("\n");

for (j = 0; j < N; j++) {

printf("%f ", Pixel[test * N + i][test * N + j]);

}

}

// 2D Discrete Cosine Transform

for (y_count = 0; y_count < y_axis_number; y_count++) {

for (x_count = 0; x_count < x_axis_number; x_count++) {

for (u = 0; u < N; u++) {

for (v = 0; v < N; v++) {

temp = 0.0;

for (i = 0; i < N; i++) {

for (j = 0; j < N; j++) {

temp += cos((2 * i + 1) * (u * M_PI) / (2 * N)) * cos((2 * j + 1) * (v * M_PI) / (2 * N)) * Pixel[y_count * N + i][x_count * N + j];

}

}

temp *= (Coeffcient[u] * Coeffcient[v]) / 4;

DCT[y_count * N + u][x_count * N + v] = temp;

}

}

}

}

// live 배열 만큼만 살린다.

for (y_count = 0; y_count < y_axis_number; y_count++) {

for (x_count = 0; x_count < x_axis_number; x_count++) {

for (u = 0; u < N; u++) {

for (v = 0; v < N; v++) {

if (u >= live || v >= live)

DCT[y_count * N + u][x_count * N + v] = 0;

}

}

}

}

// DCT 배열 출력

printf("\n\nDCT 를 적용한 %d by %d 픽셀", test + N, test + N);

for (u = 0; u < N; u++) {

printf("\n");

for (v = 0; v < N; v++) {

printf("%f ", DCT[test * N + u][test * N + v]);

}

}

// 2D Inverse Discrete Cosine Transform

for (y_count = 0; y_count < y_axis_number; y_count++) {

for (x_count = 0; x_count < x_axis_number; x_count++) {

for (i = 0; i < N; i++) {

for (j = 0; j < N; j++) {

temp = 0.0;

for (u = 0; u < N; u++) {

for (v = 0; v < N; v++) {

temp += ((Coeffcient[u] * Coeffcient[v]) / 4) * cos((2 * i + 1) * (u * M_PI) / (2 * N)) *

cos((2 * j + 1) * (v * M_PI) / (2 * N)) * DCT[y_count * N + u][x_count * N + v];

}

}

IDCT[y_count * N + i][x_count * N + j] = temp;

}

}

}

}

// IDCT 배열 출력

printf("\n\nIDCT 를 적용한 %d by %d 픽셀", test + N, test + N);

for (i = 0; i < N; i++) {

printf("\n");

for (j = 0; j < N; j++) {

printf("%f ", IDCT[test * N + i][test * N + j]);

}

}

// IDCT 배열 정수화

Mat result(height, width, CV_8U);

for (i = 0; i < height; i++) {

for (j = 0; j < width; j++) {

result.at<uchar>(i, j) = char(round(IDCT[i][j]));

}

}

// 정수화 된 IDCT 배열 출력

printf("\n\nIDCT 를 정수로 바꾼 %d by %d 픽셀", test + N, test + N);

for (i = 0; i < N; i++) {

printf("\n");

for (j = 0; j < N; j++) {

printf("%d ", result.at<uchar>(test * N + i, test * N + j));

}

}

namedWindow("result");

imshow("result", result);

waitKey(0);

return 0;

}
