# Object Detection
**신발**
<hr>

- YOLO-5 - 코드만 존재
- EfficientDet - 성능이 보장되며 논문, 코드로 존재 <- 선택

<li><b>배경에 따라 달라지는 사항이 많음</b></li>
<li>수작업으로 데이터를 만들어서 학습을 시킨 후 이용</li>
<li> AI hub 데이터 이용 Test 가능</li>

<hr>

## 목적 : 

1. Data 만들기 
2. 성능 측정
	- Region Proposal : IoU이용 성능 평가, IoU를 threshold로 설정하고 변경하면서 성능 측정
		- ex) 0.5-0.9 0.1간격.
		- confusion matrix 이용 성능 평가 확인
		- precision recall curve(넓이를 통해)

<hr>

### 과정
#### *2 stage detector*
1. bounding box(좌상, 우하) - Region proposal
2. classification
상대적으로 느린 속도

#### *1 stage detector*
- Conv Layers에서 위의 두 과정을 같이 실행, 상대적으로 속도가 빠르지만 정확성 떨어짐

[flow map참고]

- single object detection
- multi object detection

<hr>

##### Resolution(화질), size
- 보통 resolution과 size를 고정시키지만 이미지가 가까우면 resolution이 높고 멀면 resolution이 낮다
- EfficientDet을 이용해서 이미지의 **size에 제한되지 않고** 어떤 size든 가능.

---------------
> PANet, FPN 이후 
> Bi-FPN 이해
-------------

##### Compound scaling
1. depth
2. channel
3. resolution
를 모두 해결하는 방법.
