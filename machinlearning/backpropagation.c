#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>

#define InputUnitNo     2          // 입력층 노드 수 
#define HiddenUnitNo    3          // 은닉층 노드 수
#define OutputUnitNo    1           // 출력층 노드 수
#define MaxPatternNo    4          // 학습 패턴 최대수

#define Eta             0.85        // 학습률
#define Alpha           0.24         // 과거 가중치에 관한 정수
#define ErrorFunc       0.03        // 학습종료 Error
#define Wmin            -0.30       // 초기 가중치 최소값
#define Wmax            0.30        // 초기 가중치 최대값

/* Sigmoid & Random Numbers */
#define f(x) (1/(1+exp(-(x))))                              // Simoid Function
#define rnd() ((double)rand()/0x7fff * (Wmax-Wmin)+Wmin)   // Random Nubers


/* 네트워크 구성 */
double O1[MaxPatternNo][InputUnitNo];                       // 입력층 노드 출력값
double O2[HiddenUnitNo];                                    // 은닉층 노드 출력값 
double O3[OutputUnitNo];                                    // 출력층 노드 출력값
double t[MaxPatternNo][OutputUnitNo];                       // 교사신호
double W21[HiddenUnitNo][InputUnitNo];                      // 가중치 (입력층 -> 은닉층)
double dW21[HiddenUnitNo][InputUnitNo];                     // 가중치 (입력층 -> 은익층) Feed Back
double W32[OutputUnitNo][HiddenUnitNo];                     // 가중치 (은닉층 -> 출력층)
double dW32[OutputUnitNo][HiddenUnitNo];                    // 가중치 (은닉층 -> 출력층) Feed Back
double bias2[HiddenUnitNo];                                 // 은닉층 노드 활성화
double dbias2[HiddenUnitNo];                                // 은닉층 역방향 노드 활성화
double bias3[OutputUnitNo];                                 // 출력층 노드 활성화 
double dbias3[OutputUnitNo];                                // 출력층 역방향 노드 활성화
int learning_pattern_no;                                    // 학습 패턴 수
int test_pattern_no;                                        // 학습 후, 테스트 패턴 수

/* 메인 프로그램 */
void main(int argc, char* argv[])
{
    int i, j, k;                                            // 지역변수(입력, 은닉, 출력 등 반복)
    char filename[30];                                      // 데이터 파일이름
    char ss[80];                                            // 문자 처리 데이터
    double errorfunc;                                       // 학습 종료를 위한 변수

    int num = 0;                                            // Error 반복 횟수 

    /* 긴 프로그램을 단순화 하기 위한 함수로 분리 */
    void propagation();                                     // 순방향 네트워크 함수
    void back_propagation();                                // 역방향 네트워크 함수
    void state();                                           // 출력값 표시 함수
    void read_file();                                       // 파일 읽기 함수
    void initialize();                                      // 가중치 초기화 함수

    printf("Learning File : ");
    scanf("%s", &filename);                                
    printf("\n # 입력한 파일명: %s", filename);

    /* 가중치변화 파일 별도 저장*/
    FILE* fp = fopen("ErrorFunc.txt", "w");

    read_file(filename);                         
    initialize();       

    printf("\n\n ****************** 학습하기 전 ******************\n");
    printf("\tPattern Output1 Output2 Output3 Output4\n");

    /* 학습 패턴에 대한 학습 및 결과값 표시*/
    for (errorfunc = 0.0, i = 0; i < learning_pattern_no; i++) {
        state(i);
        for (j = 0; j < OutputUnitNo; j++)
            errorfunc += pow(t[i][j] - O3[j], 2.0);
    }
    errorfunc /= 2;
    printf(" ErrorFunc : %.3f\n", errorfunc);

    /* 학습 시작 */
    printf("\n******************* 학습 시작 ******************* ");
    printf("\nCount Pattern  Output1  Output2  Output3  Output4\n");
    for (i = 0; errorfunc > ErrorFunc; )                                 // Error 값 이하 학습종료
    {
        for (j = 0; j < learning_pattern_no; j++){                      
            propagation(j);                                             
            back_propagation(j);                                        
        }
        for (errorfunc = 0.0, j = 0; j < learning_pattern_no; j++){      
            printf("%d", ++i);
            state(j);                                                   
            for (k = 0; k < OutputUnitNo; k++)                         // 출력 노드 수 
             errorfunc += pow(t[j][k] - O3[k], 2.0);                 // Error값 산출
        }
        ++num;
        errorfunc /= 2;
        printf("ErrorFunc : %.3f\n", errorfunc);                      
        fprintf(fp, "%d : %lf \n", num, errorfunc);                      
    }
    fclose(fp);                                                        

    /* 학습 종료 */
    printf("\n *************** 신규 데이터 결과 ***************\n");
    printf("\tPattern Output1 Output2 Output3 Output4\n");

    /* 신규 데이터 예측 결과 */
    for (i = learning_pattern_no; i < learning_pattern_no + test_pattern_no; i++)       
        state(i);
}

/*순방향*/
void propagation(p)
int p;
{
    int i, j;
    double net;

    /* 은닉층 노드에서 출력 계산 */
    for (i = 0; i < HiddenUnitNo; i++){
        for (net = 0.0, j = 0; j < InputUnitNo; j++) {
            net += W21[i][j] * O1[p][j];            // W21 가중치와 O1의 합
        }         
         O2[i] = f(net + bias2[i]);                                      // O2 출력값 = 활성화함수 f()
    }

    /* 출력 노드에서 출력 계산 */
    for (size_t i = 0; i < OutputUnitNo; i++)
    {
        for (net = 0.0, j = 0; j < HiddenUnitNo; j++) {
            net += W32[i][j] * O2[j];                               // W32 가중치와 O2의 합
        }
        O3[i] = f(net + bias3[i]);                                  // O3 출력값 = 활성화 함수 f()
    }
}

/* 역방향 */
int p;
void back_propagation(p){
    int i, j;
    double d2[HiddenUnitNo];                                            // 가중치 변경 입력 -> 은닉
    double d3[OutputUnitNo];                                            // 가중치 변경 은닉 -> 출력
    double sum;                                                         // 합를 구하기 위한 변수

/* 평균제곱오차: dj=(oj-yj) f'(i,j) */
    for (i = 0; i < OutputUnitNo; i++) {
        d3[i] = (t[p][i] - O3[i]) * O3[i] * (1 - O3[i]);     // 출력 오차 값
    }                   

    /* 가중치 변경  w_(i,j) =  ηd_(j) O_(i)*/
    for (i = 0; i < HiddenUnitNo; i++)
    {
        for (sum = 0.0, j = 0; j < OutputUnitNo; j++)
        {
            dW32[j][i] = Eta * d3[j] * O2[i] + Alpha * dW32[j][i];      // Eta 학습률과 역방향 가중치에 Alpha 학습률 추가
            W32[j][i] += dW32[j][i];                                    // 역방향 학습 가중치를 순방향 가중치로 변환 
            sum += d3[j] * W32[j][i];                                   // 합 = 출력층 바이어스와 가중치(출력층->은닉층)
        }
        /* d_(j) = (Σ w_(j,i) d_(i)) f'(i,j) */
        d2[i] = O2[i] * (1 - O2[i]) * sum;
    }
    for (i = 0; i < OutputUnitNo; i++)
    {
        dbias3[i] = Eta * d3[i] + Alpha * dbias3[i];                    // Eta 학습률과 역방향 가중치에 Alpha 학습률 추가
        bias3[i] += dbias3[i];                                          // 출력 노드 활성화
    }

    /* ?w_(i,j) =  ηd_(j) O_(i) */
    for (i = 0; i < InputUnitNo; i++)
    {
        for (j = 0; j < HiddenUnitNo; j++)
        {
            dW21[j][i] = Eta * d2[j] * O1[p][i] + Alpha * dW21[j][i];   // 역방향 은닉층 가중치
            W21[j][i] += dW21[j][i];                                    // W21 가중치
        }
    }
    for (i = 0; i < HiddenUnitNo; i++)
    {
        dbias2[i] = Eta * d2[i] + Alpha * dbias2[i];                    // 역 방향 은닉층 노드 활성화
        bias2[i] += dbias2[i];                                          // 은닉층 노드 활성화
    }
}

void state(int p){
    int i;
    printf("\t%d -> ", p + 1);                                            // state()함수에 삽입되는 p값 + 1 값 
    propagation(p);
    for (i = 0; i < OutputUnitNo; i++)
    {
        printf("\t%5.3f", O3[i]);                                    
    }
    fputs("\n", stdout);
}

void read_file(char* name){
    int i, j;
    FILE* fp;

    if ((fp = fopen(name, "r")) == NULL) {
        fprintf(stderr, "\n%s : File Open Error !!\n", name);
        exit(-1);
    }

    fscanf(fp, "%d", &learning_pattern_no);
    for (i = 0; i < learning_pattern_no; i++) {
        for (j = 0; j < InputUnitNo; j++) {
            fscanf(fp, "%lf", &O1[i][j]); }
        for (j = 0; j < OutputUnitNo; j++) {
            fscanf(fp, "%lf", &t[i][j]);  }
    }

    /* 테스트 데이터 읽어 오기 */
    fscanf(fp, "%d", &test_pattern_no);
    for (i = learning_pattern_no; i < learning_pattern_no + test_pattern_no; i++) {
        for (j = 0; j < InputUnitNo; j++) {
            fscanf(fp, "%lf", &O1[i][j]); }
    }
    fclose(fp);
}

/* 가중치 초기화 */
void initialize(){
    int i, j;

    /* 가중치 초기화 입력층 -> 은닉층 */
    for (i = 0; i < HiddenUnitNo; i++) {
        for (j = 0; j < InputUnitNo; j++) {
            W21[i][j] = ((double)rand() / 0x7fff * (Wmax - Wmin) + Wmin); }
    }

    /* 가중치 초기화 은닉층 -> 출력층 */
    for (i = 0; i < OutputUnitNo; i++)  {
        for (j = 0; j < HiddenUnitNo; j++) {
            W32[i][j] = ((double)rand() / 0x7fff * (Wmax - Wmin) + Wmin); }
    }
}
