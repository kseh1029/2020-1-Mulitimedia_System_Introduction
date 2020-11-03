# HW 3 : Laplacian Filtering

#### 과제에 대한 설명

◦ 수업에서 배운 Laplacian filtering 을 C/C++/Java/Python 중에서 구현

◦ 영상 입력, 출력, 필터링은 OpenCV 사용

◦ 구현 순서

1) 입력영상을 read 한다.

2) 영상을 흑백으로 변환한다.

3) Gaussian smoothing 으로 영상을 blur 한다. (잡음도 제거)

void cv::GaussianBlur();

4) Blur된 영상에 Laplacian filtering 적용한다.

void cv::Laplacian();

- 주의점: Laplace 연산은 값이 255을 넘기때문에 영상 type을 interger

로 사용 바람. (float으로 하면 image display가 안되어 다시 변환필요)

![HW3_How%20e22160036cc74acfab7e446e445a99aa/picture10.jpeg](HW3_How%20e22160036cc74acfab7e446e445a99aa/picture10.jpeg)

추가 고찰

1) 컬러영상을 그대로 Gaussian 과 Laplacian 적용하면 결과는?

2) 도로영상이 결과에서 직선 차선을 찾을 수 있는 방안은?

- 직선 방정식을 이용.
- 인터넷에 많은 내용이 있으니 찾아보자.





Ⅰ. 과제 결과

1) lena.jpg

**Input Image**

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image4.bmp](HW3%207920259c7c4c4aa1a3119808a6543ffd/image4.bmp)

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image5.bmp](HW3%207920259c7c4c4aa1a3119808a6543ffd/image5.bmp)

**Grayscaled Image**

[Untitled](HW3%207920259c7c4c4aa1a3119808a6543ffd/Untitled%20Database%200b6d74bf6446406a96cf0e0f53a951e8.csv)

2) juniel.jpg

**Input Image**

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image21.bmp](HW3%207920259c7c4c4aa1a3119808a6543ffd/image21.bmp)

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image22.bmp](HW3%207920259c7c4c4aa1a3119808a6543ffd/image22.bmp)

**Grayscaled Image**

[Untitled](HW3%207920259c7c4c4aa1a3119808a6543ffd/Untitled%20Database%20e10a3d3a213a42f485de9955e8de8de6.csv)

3) road.jpg

**Input Image**

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image38.bmp](HW3%207920259c7c4c4aa1a3119808a6543ffd/image38.bmp)

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image39.bmp](HW3%207920259c7c4c4aa1a3119808a6543ffd/image39.bmp)

**Grayscaled Image**

[Untitled](HW3%207920259c7c4c4aa1a3119808a6543ffd/Untitled%20Database%20ce71dcdd50b0413088470f7b28869735.csv)

Ⅱ. HW#2 목적, 접근방법 및 세부결과

해당 HW #3 의 Laplace Filtering 는 edge detection 하는게 목적이다.

24 bit RGB 이미지를 8 bit Grayscale 로 변환하고 변환된 사진을 Gaussian Smoothing 후 Laplacian Filtering을 진행하였다.

마지막으로 Zero crossing을 구하여 Edge detection을 하는데 여러번의 시행차고를 겪었는데 이는 Mat Class Structrue 에 대한 이해도가 부족했기 때문이였다.

따라서 이 과제를 진행하기에 앞서 여러 가지 학습해야할 내용이 있었다.

**① 기본 Data Type**

> CV_8U : 8-bit unsigned integer: uchar ( 0..255 )CV_8S : 8-bit signed integer: schar ( -128..127 )CV_16U : 16-bit unsigned integer: ushort ( 0..65535 )CV_16S : 16-bit signed integer: short ( -32768..32767 )CV_32S : 32-bit signed integer: int ( -2147483648..2147483647 )CV_32F : 32-bit floating-point number: float ( -FLT_MAX..FLT_MAX, INF, NAN )CV_64F : 64-bit floating-point number: double ( -DBL_MAX..DBL_MAX, INF, NAN )

**② Mat Class structure**

RGB 이미지를 처음 imread 하고 나면 영상은 CV_8UC3 (unsigned 8bit/pixel) type 일 것이다. 대부분의 이미지는 일반적으로 R,G,B를 각각 8bits 로 지정하기 때문이다.

grayscale 로 이미지를 변환하고 나면 1개의 채널만 사용하게 되므로 CV_8U type 이 된다.

일반적으로 CV_8U를 통해 각 픽셀의 0~255 의 숫자를 다뤄 해결할 수 있지만 수학적인 계산을 할때에는 8bit 로만 해결할 수 없었다.

[Untitled](HW3%207920259c7c4c4aa1a3119808a6543ffd/Untitled%20Database%20f968e9d7aad7432aaeda9f8740c2e51a.csv)

해당 내용은 stack overflow에서 찾아볼 수 있었으며 다음과 같이 설명되어 있었다.

즉, 데이터의 계산이나 수학적인 툴을 사용하기 위해서 CV_32F 혹은 CV_32S 로 변환해야 한다는 것이다.

연산이 끝나고 나서 이 결과를 실제 이미지로 사용하기 위해서는 다시 8bit 이미지로 변환해야지 저장이나 디스플레이를 할 수가 있다.

Zerocrossing 함수는 수학적인 내용을 다루는 함수이므로 입력을 32bit 로 변환해줄 필요가 있다. 그렇지 않으면 예상치 못한 에러가 발생하거나 올바른 연산이 되지 않는다.

Mat Array의 비트를 변환하는 명령어는 다음과 같다.

> void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 )여기서 OutputArray 는 출력 Mat를 뜻하고, rtype 은 bit type 이다.alpha 는 value 에 얼마를 곱해주고 출력을 할 것인가를 설정한다.beta 는 value 에 얼마를 더해주고 출력을 할 것인가를 설정한다.이를 실제로 적용해서 8bit Graycaled image를 32Bit Float 로 변환하기 위해서는 다음와 같이 코드를 사용할 수 있다.Grayscale_8U.convertTo(Grayscale_32F, CV_32F, 1.0/255.0);

여기서 1.0/255.0을 곱해준 이유는 Float 로 변환하였으므로 0~1.0 사이의 data range를 가지게 하기 위해서이다.

beta 는 생략하면 0이 기본적으로 사용된다.

**③ Laplacian Filtering 이론**

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image55.jpeg](HW3%207920259c7c4c4aa1a3119808a6543ffd/image55.jpeg)

1차원 기준으로 밝기 다이어그램을 볼 때, 밝기가 가장 급격하게 변하는 부분을 Edge 로 사용할 수 있을 것이다.

Sobel Edge Eetection 은 x와 y에 대해 각각 한번씩만 편미분을 하여 크기를 구해 Edge를 찾았지만

Laplacian 은 2번 미분하여 edge를 찾는 방식이다.

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image56.jpeg](HW3%207920259c7c4c4aa1a3119808a6543ffd/image56.jpeg)

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image57.png](HW3%207920259c7c4c4aa1a3119808a6543ffd/image57.png)

위의 그림처럼 2차 미분값이 0인 부분을 찾는 것이다.

하지만, 조심해야 할 점이 있다.

2번을 미분해서 0이 나온 값이아닌 원래부터 0이였던 값을 Edge 로 인식할 수 있다.

그래서 단순히 0을 찾는게 아닌 부호가 바뀌는 지점 사이의 0을 Edge로 사용하여야 한다.

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image58.jpeg](HW3%207920259c7c4c4aa1a3119808a6543ffd/image58.jpeg)

하지만, Noise 로 인해서 위의 그림처럼 엉뚱한 부분을 Edge로 인식할 수 있다.

그래서 이 Noise를 감소시키기 위하여 Blur 처리를 하고, Laplacian을 적용시키는 것이다.

Blur를 하기 위해서는 여러 가지 방법이 있는데, 그 중에서 Gaussian Blur를 사용한다.

그리고 나서 Laplacian Filter를 적용하고 Zero Detection을 한다면, Edge를 얻을 수 있게 된다.

이를 통틀어 Laplacian of Gaussian (LoG) 라는 명칭으로 불린다.

실제로 사용할 때처럼 순서대로 정리해보면 다음과 같다.

$G(x,\, y) = e^{- \frac{x^{2} + y^{2}}{2\sigma^{2}}}$ : 2D Gaussian Function (표준편차 시그마가 커질수록 이미지가 흐려진다)

=> 2D 가우시안 함수를 사용하여 이미지를 블러시킨다.

∇2 즉 $\frac{\partial^{2}}{\partial x^{2}} + \frac{\partial^{2}}{\partial y^{2}}$ : Laplacian 연산자 (수평 방향, 수직 방향으로의 2차 미분을 구한다)

이를 하나로 통합시 $\nabla^{2}G(x,y) = \frac{\partial^{2}G(x,y)}{\partial x^{2}} + \frac{\partial^{2}G(x,y)}{\partial y^{2}}$

$\nabla^{2}G(x,y) = \frac{\partial}{\partial x}\lbrack\frac{- x}{\sigma^{2}}e^{- \frac{x^{2} + y^{2}}{2\sigma^{2}}}\rbrack + \frac{\partial}{\partial y}\lbrack\frac{- y}{\sigma^{2}}e^{- \frac{x^{2} + y^{2}}{2\sigma^{2}}}\rbrack$

$\nabla^{2}G(x,y) = \lbrack\frac{x^{2}}{\sigma^{4}} - \frac{1}{\sigma^{2}}\rbrack e^{- \frac{x^{2} + y^{2}}{2\sigma^{2}}} + \lbrack\frac{y^{2}}{\sigma^{4}} - \frac{1}{\sigma^{2}}\rbrack e^{- \frac{x^{2} + y^{2}}{2\sigma^{2}}}$

$\nabla^{2}G(x,y) = \lbrack\frac{x^{2} + y^{2} - 2\sigma^{2}}{\sigma^{4}}\rbrack e^{- \frac{x^{2} + y^{2}}{2\sigma^{2}}}$ : Laplacian of Gaussian 연산자

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image59.bmp](HW3%207920259c7c4c4aa1a3119808a6543ffd/image59.bmp)

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image60.png](HW3%207920259c7c4c4aa1a3119808a6543ffd/image60.png)

> Laplacian Mask 는 정해져있는 것은 아니다. n이 홀수이어야 하고, 모든 element 의 값을 더했을 때 0이 되고, 위의 그림과 같은꼴만 나오면 된다.모든 element 값의 합이 0이 되어야 하는 이유는 픽셀값의 변화가 없는 부분에서는 반응하지 않아야 하기 때문이다.

**④ OpenCV에서 사용하는 방법**

OpenCV에서 Gaussian Blur를 사용하는 방법이다.

> GaussianBlur(InputArray, OutputArray, kisze, sigmaX, sigmaY, border_type);InputArray 는 입력 Mat, OutputArray 는 출력 Mat를 입력하면 된다.단, 이때 InputArray 는 반드시 CV_8U, CV_16U, CV_16S, CV_32F, CV_64F 중 하나여야 한다.ksize 에는 Gaussian kernel size를 입력하면 되는데, 반드시 양수이면서 홀수를 입력해야 한다.예를 들어 5x5 의 Gaussian Blur를 하고싶으면 ksize 자리에 Size(5,5)를 입력하면 된다.sigmaX 에는 X 방향으로의 표준편차 시그마를 입력하면 된다.sigmaY 에는 Y 방향으로의 표준편차 시그마를 입력하면 되는데, 입력하지 않거나 0을 입력하면 sigmaX 와 동일하게 설정된다.borderType 은 Laplacian Filter 에 설명한 것과 동일하다.

OpenCV에서 Laplacian Filter를 사용하는 방법이다.

> Laplacian(InputArray, OutputArray, ddepth, kernel_size, scale, delta, border_type);InputArray 는 입력 Mat, OutputArray 는 출력 Mat를 입력하면 된다.ddepth 는 OutputArray 로 뱉어낼 이미지의 type을 입력한다. 추후 Zerocrossing 함수에서 입력을 CV_32F 로 받으므로 CV_32F 로 적는 것이 좋다.kerner_size 는 2차 미분 필터의 크기를 설정하며 1, 3, 5, 7처럼 홀수를 사용한다.만약 kerner_size 가 1일 경우 3x3 Aperture Size를 사용하여 다음과 같은 필터가 된다.

[Untitled](HW3%207920259c7c4c4aa1a3119808a6543ffd/Untitled%20Database%20e9dc8085931047d48a9284282f5ae39f.csv)

> scale 은 계산된 미분 값에 대한 배율 값이다.delta 는 계산전 미분 값에 대한 추가 값이다.border_type 은 픽셀 외삽법으로 이미지를 가장자리 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당해야 하는데, 이 때 사용하는 테두리 설정이다.픽셀 외삽법의 종류는 다음과 같다

[Untitled](HW3%207920259c7c4c4aa1a3119808a6543ffd/Untitled%20Database%2049c9f5c34d8f4d2a98bf3ff84c9cd175.csv)

Ⅲ. 고찰

도로영상이 결과에서 직선 차선을 찾을 수 있는 방안은?

=> 1) Hough Transform을 사용 (가장 대표적인 방법, 연산량 부담이 크다)

=> 2) 차선 영역의 밝기 특성을 활용하여 차선색상 분석으로 관심영역 (ROI) 으로 설정하여 한정된 범위 탐색

(연산 속도의 향상) 후 Spline 혹은 RANSAC 등의 알고리즘을 활용해 3차 이상의 함수로 차선 방정식 도출

=> 3) 생체 모방 비전 센서 기반의 차선 인식 방법으로 인간의 뇌를 구성하고 인식을 담당하는 뉴런들을 하드웨어 형태로 구현 및 집적하여 차선의 형태를 학습 및 인식할 수 있도록 하고, 이를 비전 센서와 결합하여 실시간으로 입력되는 영상 신호에서 사전에 학습된 차선 패턴을 인식하여 분류

<직선 차선을 찾는 가장 대표적인 방법>

먼저 RGB 영상을 입력받는다.

Grayscale 로 영상을 변환하고, Blur를 하여 Noise를 감소시킨다.

Edge Detection 방법을 사용해 edge를 검출한다.

이 edge 중 직선을 나타내는 윤곽을 찾기 위해 Hough transform을 사용한다.

**ⓞ Hough transform**

Hough transform 은 이미지상에서 직선, 원 등의 특정 모양을 찾는 과정이다.

이미지 상의 특정한 점들간의 연관성을 찾아 특징을 추출하는 방법으로 원리는 다음과 같다.

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image61.png](HW3%207920259c7c4c4aa1a3119808a6543ffd/image61.png)

왼쪽의 xy 평면에서 기울기는 *a*1 , y 절편은 *b*1 을 가지는 *y* = *a*1*x* + *b*1 이 있다고 할 때

이 직선위에 (*y*1, *x*1), (*y*2, *x*2), (*y*3, *x*3) 가 있다고 할 때

이 점들을 xy 평면 상에서 기울기, y절편의 평면인 a,b 평면으로 옮기게 되면 각각 하나의 직선이 나와

총 세 개의 직선이 나오게 된다.

즉 일반적으로 *y* = ax + *b* 의 방정식이 *b* = − ax + *y* 로 변환되는 것이다.

(*y*1, *x*1), (*y*2, *x*2), (*y*3, *x*3) 각각의 점을 대입하게 되면

*b* = − *a*1*x* + *y*1, *b* = − *a*2*x* + *y*2, *b* = − *a*3*x* + *y*3 라는 세 개의 식이 나온다.

이 때 ab 평면이 직선의 기울기와 y절편이라는 것을 고려했을 때

같은 직선 상의 점들은 당연히 같은 기울기와 y절편을 가지므로 교점을 형성한다.

지금은 같은 직선 상에서 임의이 세 점을 선택했으므로 ab 평면에서 세 직선이 하나의 교점을 갖고

그 좌표 값은 (*a*1, *b*1) 을 가지게 된다.

이 좌표값을 가지고 ab 평면에서 다시 xy 평면으로 바꾸게 되면 기울기와 y절편을 알고 있으므로 하나의 직선을 구할 수 있게 된다.

이런 원리로 xy 평면 상에서 같은 직선인지 아닌지 모르는 임의의 점들을 ab평면으로 매핑하고,

ab평면에서 직선들간의 교점 존재 여부를 확인하면 같은 직선 상의 점인지 아닌지를 확인할 수 있다.

하지만 이런 방법에는 한가지 문제점이 생기게 된다.

만약 기울기가 0일 경우 무수히 많은 직선이 형성되어 각 점들간의 연관성을 찾을 수 없게 된다.

![HW3%207920259c7c4c4aa1a3119808a6543ffd/image62.jpeg](HW3%207920259c7c4c4aa1a3119808a6543ffd/image62.jpeg)

*r*

*θ*

로 변환한다.

xy 평면에서의 직선을 *r* 과 *θ* 에 대한 표현으로 바꿀 수 있다.

원점에서부터 직선까지의 거리를 *r*, x 축과의 기울기를 *θ* 라고 하면

$a = \frac{B}{A} = - \frac{\sin\theta}{\cos\theta}$, $b = \frac{r}{\sin\theta}$ 이 되므로

$y = ( - \frac{\sin\theta}{\cos\theta})x + \frac{r}{\sin\theta}$

*r* = xcos*θ* + ysin*θ* 이다.

Ⅳ. C++ 코드

/* 2017117876 김승현 HW#3

Laplacian Filtering */

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

void FindZeroCrossings(Mat& laplacian, Mat& zero_crossings) {

Mat* result = new Mat(laplacian.size(), CV_8U, Scalar(0));

zero_crossings = *result;

int image_rows = laplacian.rows;

int image_channels = laplacian.channels();

int values_on_each_row = laplacian.cols * image_channels;

float laplacian_threshold = 0.0;

// Find Zero Crossings

for (int row = 1; row < image_rows; row++) {

float* prev_row_pixel = laplacian.ptr<float>(row - 1) + 1;

float* curr_row_pixel = laplacian.ptr<float>(row);

uchar* output_pixel = zero_crossings.ptr<uchar>(row) + 1;

for (int column = 1; column < values_on_each_row; column++)

{

float prev_value_on_row = *curr_row_pixel;

curr_row_pixel++;

float curr_value = *curr_row_pixel;

float prev_value_on_column = *prev_row_pixel;

float difference = 0.0;

if (((curr_value > 0) && (prev_value_on_row < 0)) ||

((curr_value < 0) && (prev_value_on_row > 0)))

difference = abs(curr_value - prev_value_on_row);

if ((((curr_value > 0) && (prev_value_on_column < 0)) ||

((curr_value < 0) && (prev_value_on_column > 0))) &&

(abs(curr_value - prev_value_on_column) > difference))

difference = abs(curr_value - prev_value_on_column);

- output_pixel = (difference > laplacian_threshold) ? 255 : 0;// (int) ((100 * difference) / laplacian_threshold);

prev_row_pixel++;

output_pixel++;

}

}

}

int main(void) {

Mat input = imread("lena.jpg", IMREAD_COLOR);

if (input.data == NULL) {

std::cout << "해당 사진이 없습니다." << std::endl;

return -1;

}

Mat Grayscale_32F, Grayscale_8U;

cvtColor(input, Grayscale_8U, COLOR_RGB2GRAY);

Grayscale_8U.convertTo(Grayscale_32F, CV_32F, 1.0 / 255.0);

namedWindow("Input Image");

imshow("Input Image", input);

namedWindow("grayscale_32F");

imshow("grayscale_32F", Grayscale_32F);

Mat blurred_Grayscale_32F[5];

GaussianBlur(Grayscale_32F, blurred_Grayscale_32F[0], Size(5, 5), 1);

GaussianBlur(Grayscale_32F, blurred_Grayscale_32F[1], Size(5, 5), 2);

GaussianBlur(Grayscale_32F, blurred_Grayscale_32F[2], Size(5, 5), 3);

GaussianBlur(Grayscale_32F, blurred_Grayscale_32F[3], Size(5, 5), 4);

GaussianBlur(Grayscale_32F, blurred_Grayscale_32F[4], Size(5, 5), 5);

namedWindow("Gaussian Blur (sigma=1)");

imshow("Gaussian Blur (sigma=1)", blurred_Grayscale_32F[0]);

namedWindow("Gaussian Blur (sigma=2)");

imshow("Gaussian Blur (sigma=2)", blurred_Grayscale_32F[1]);

namedWindow("Gaussian Blur (sigma=3)");

imshow("Gaussian Blur (sigma=3)", blurred_Grayscale_32F[2]);

namedWindow("Gaussian Blur (sigma=4)");

imshow("Gaussian Blur (sigma=4)", blurred_Grayscale_32F[3]);

namedWindow("Gaussian Blur (sigma=5)");

imshow("Gaussian Blur (sigma=5)", blurred_Grayscale_32F[4]);

Mat laplacian_32F[5];

Laplacian(blurred_Grayscale_32F[0], laplacian_32F[0], CV_32F, 3, 5, 0.3);

Laplacian(blurred_Grayscale_32F[1], laplacian_32F[1], CV_32F, 3, 5, 0.3);

Laplacian(blurred_Grayscale_32F[2], laplacian_32F[2], CV_32F, 3, 5, 0.3);

Laplacian(blurred_Grayscale_32F[3], laplacian_32F[3], CV_32F, 3, 5, 0.3);

Laplacian(blurred_Grayscale_32F[4], laplacian_32F[4], CV_32F, 3, 5, 0.3);

namedWindow("Laplacian (sigma=1)");

imshow("Laplacian (sigma=1)", laplacian_32F[0]);

namedWindow("Laplacian (sigma=2)");

imshow("Laplacian (sigma=2)", laplacian_32F[1]);

namedWindow("Laplacian (sigma=3)");

imshow("Laplacian (sigma=3)", laplacian_32F[2]);

namedWindow("Laplacian (sigma=4)");

imshow("Laplacian (sigma=4)", laplacian_32F[3]);

namedWindow("Laplacian (sigma=5)");

imshow("Laplacian (sigma=5)", laplacian_32F[4]);

Mat zero_crossings_8U[5];

FindZeroCrossings(laplacian_32F[0], zero_crossings_8U[0]);

FindZeroCrossings(laplacian_32F[1], zero_crossings_8U[1]);

FindZeroCrossings(laplacian_32F[2], zero_crossings_8U[2]);

FindZeroCrossings(laplacian_32F[3], zero_crossings_8U[3]);

FindZeroCrossings(laplacian_32F[4], zero_crossings_8U[4]);

namedWindow("Zero Crossing (sigma=1)");

imshow("Zero Crossing (sigma=1)", zero_crossings_8U[0]);

namedWindow("Zero Crossing (sigma=2)");

imshow("Zero Crossing (sigma=2)", zero_crossings_8U[1]);

namedWindow("Zero Crossing (sigma=3)");

imshow("Zero Crossing (sigma=3)", zero_crossings_8U[2]);

namedWindow("Zero Crossing (sigma=4)");

imshow("Zero Crossing (sigma=4)", zero_crossings_8U[3]);

namedWindow("Zero Crossing (sigma=5)");

imshow("Zero Crossing (sigma=5)", zero_crossings_8U[4]);

imwrite("zero_crossings_sigma[0].jpg", zero_crossings_8U[0]);

imwrite("zero_crossings_sigma[1].jpg", zero_crossings_8U[1]);

imwrite("zero_crossings_sigma[2].jpg", zero_crossings_8U[2]);

imwrite("zero_crossings_sigma[3].jpg", zero_crossings_8U[3]);

imwrite("zero_crossings_sigma[4].jpg", zero_crossings_8U[4]);

waitKey(0);

return 0;

}

Ⅴ. 참고문헌

**OpenCV Laplace Operator**

[https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html)

**Python OpenCV 강좌 : 제 14강 - 가장자리 검출**

[https://076923.github.io/posts/Python-opencv-14/](https://076923.github.io/posts/Python-opencv-14/)

**Laplacian of Gaussian (LoG), 엣지 검출의 한 방법**

[https://bskyvision.com/133](https://bskyvision.com/133)

**[Udacity] SelfDrivingCar- 2-3. 차선 인식(hough transform)**

[https://m.blog.naver.com/windowsub0406/220894462409](https://m.blog.naver.com/windowsub0406/220894462409)

**생체 모방 비전 센서 기반의 차선 검출 기법 (2017 한국자동차공학회 춘계학술대회, 만도헬라일렉트로닉스 신사업개발팀)**

[https://www.ksae.org/func/download_journal.php?path=L2hvbWUvdmlydHVhbC9rc2FlL2h0ZG9jcy91cGxvYWQvam91cm5hbC9BYnN0cmFjdF8xNTY1ODM2NDA2XzI2MDYucGRm&filename=S1NBRTE3LVMwMjkwLnBkZg==&bsid=26814](https://www.ksae.org/func/download_journal.php/?path=L2hvbWUvdmlydHVhbC9rc2FlL2h0ZG9jcy91cGxvYWQvam91cm5hbC9BYnN0cmFjdF8xNTY1ODM2NDA2XzI2MDYucGRm&filename=S1NBRTE3LVMwMjkwLnBkZg==&bsid=26814)