# HW1 : DSI (Display Serial Interface)

**ⓞ MIPI 의 대표적인 인터페이스**

부품 및 기술의 표준화를 통해 제품 제조의 유연성 및 호환성을 가져다 줄 수 있다.

이와 같은 목적으로 ARM, Intel, Nokia, 삼성, STMicroelectronics, Texas Instrument 가 모여 MIPI를 설립했다.

이 MIPI에서 만든 대표적인 인터페이스는 CSI 와 DSI 이다.

이중 **DSI 는 Display Serial Interface 로 Host Processor 와 Display Module 사이에 고속 직렬 인터페이스 표준**이다.

![HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image0.bmp](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image0.bmp)

OSI 7 Layer Model에서 Physical Layer 에서는 C-PHY, D-PHY, M-PHY 가 있고 Protocal Layer 에서는 CSI, DSI 가 있다.

![HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image1.bmp](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image1.bmp)

CSI 와 DSI 의 연결도를 보면 두 가지 차이점을 확인할 수 있다.

첫 번째로 Data의 전송 방향이 다르다.

DSI 의 Data 신호는 AP에서 디스플레이로 보내는 방식이고 CSI 의 Data 신호는 카메라에서 AP로 보내는 방식이다.

두 번째로 Control Interface는 CSI에만 존재한다.

CSI 의 경우 카메라를 제어하기 위한 인터페이스로 IIC를 사용하지만 DSI 의 경우 Control Inferface가 존재하지 않는다.

**ⓞ 연도별 Physical Layer 특성**

[Untitled](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/Untitled%20Database%209934836ba5334797ad139b0a6f96bb6f.csv)

**ⓞ DSI의 종류**

DSI 는 MIPI에서 Multipedia Protocols 중에 Display and Touch 로 구분이 되어있다.

이중 Display는 크게 MIPI DSI, MIPI DSI-2 로 나눠져 있다.

DSI 는 스마트폰, 테블릿, 노트북, 자동차 및 기타플랫폼의 디스플레이를 위한 다용도 고속 인터페이스이다.

DSI-2 는 D-PHY 및 C-PHY 모두에 대한 확장성과 지원을 제공하는 미래 지향적인 고속 인터페이스이다.

DSI 는 D-PHY 물리 계층에서 작동하며 MIPI DCS(MIPI Display Command Set)에 정의된 명령어들을 사용한다.

또한 VESA(Video Electronics Standards Association)의 디스플레이 스트림 압축(DSC) 표준을 사용한다.

전반적으로 DSI의 명령어는 D-PHY 및 C-PHY를 모두 지원하는 최신 DSI-2 와 비슷하다.

[Untitled](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/Untitled%20Database%20dda0df047ca046c0919985253771a6e5.csv)

이처럼 DSI 은 D-PHY 의 다양한 버전을 지원한다.

DSI 프로토콜은 Physical Layer을 기반으로 성능이 차이난다.

> D-PHY, M-PHY, C-PHY를 비교해보면 다음과 같다.

[Untitled](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/Untitled%20Database%20e4cabb2358ae486abe199b088389bac9.csv)

**ⓞ D-PHY 와 C-PHY 의 차이점**

현재까지 사용되고 있는 CSI 와 DSI 는 대부분 MIPI D-PHY 규격을 사용하고 있지만 Bandwidth 가 늘어나면서 C-PHY 규격이 새롭게 추가 되었다.
D-PHY 와 C-PHY 의 가장 큰 차이점은 연결 방식이다.

기존의 D-PHY는 1개의 Clock lane, 최대 4개의 Data lane 으로 구성되어 있고 1 lane 당 2개의 Pin이 필요하여 총 필요한 Pin 의 수가 10개이다.

[Untitled](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/Untitled%20Database%201eed6215b540427292f3b89b20de84ca.csv)

> 차동신호방식을 기본으로 하고 그 위에 신호를 확장하는 방식이다.0.2V의 공통모드, 0.2V의 차분신호 1Gb/s 로 2개의 전송선을 통해 전송한다.저전력(LP, Low-Power) 모드에서는 1.2V 차동 LVCMOS (Low-Voltage CMOS) 전송 방식을 사용하고 10Mbps 의 단일 출력 신호로 2개의 라인을 통해 각각 전송된다.

C-PHY는 최대 3개의 Data lane 만이 있으며, 1 lane 당 3개의 Pin이 사용된다. Clock lane은 없으며 Data 에 Embedded Clock을 사용하며 Data 수신단에서 Clock Recovery를 수행하여 사용한다.

3개의 Pin을 사용하므로 단순히 High, Low 로만 사용하지 않고 High, Middle, Low 로 3가지 상태를 나타낼 수 있다. 총 3개의 pin이 있으므로 27가지의 상태를 나타낼 수 있지만, 실제로는 차분 신호로의 특성을 살릴 수 있는 6개의 상태를 사용한다.

전송속도를 클럭이 아닌 초당 심볼수로 나타내며 심볼당 데이터는 2.28bit 이다.

lane 당 2.5Gsymbol/s 가 가능하므로 총 5.7Gb/s 가 되어, 4개의 lane이면 22.8Gb/s 이므로 D-PHY 2.0 가

4 lane 일 때 속도인 18 Gb/s 보다 빠르다.

**ⓞ DSI 적용 사례**

라즈베리파이는 디스플레이가 없는 사용하는 마이크로프로세서이다. MIPI DSI Interface를 이용하여 소형 모니터와 마이크로프로세서를 연결하여 AP에서 디스플레이로 화면을 송출한다.

뿐만 아니라 이를 활용해서 사용하지 않는 휴대폰의 디스플레이를 출력장치로 사용하여 연결할 수 있다.

![HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image2.jpeg](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image2.jpeg)

![HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image3.jpeg](HW1%207f6bb6f43c7d42938b61126f24a4d0c0/image3.jpeg)

**ⓞ 참고자료**

> Control Smartphone Display from MIPI Display Serial Interfacehttps://blog.adafruit.com/2016/03/31/control-smartphone-display-from-mipi-display-serial-interface-arduino-and-raspberrypi-compatible/MIPI Display Serial Interface (MIPI DSI)https://www.mipi.org/specifications/dsiSpecification for Display Serial Interface (DSI)http://bfiles.chinaaet.com/justlxy/blog/20171114/1000019445-6364627609238902374892404.pdfMobile Industry Processor Interface(MIPI)의 D-PHY에 대해 알아볼까요?https://laonple.blog.me/220875095385Mobile Industry Processor Interface(MIPI)의 C-PHY에 대해 알아볼까요?https://laonple.blog.me/220876151247