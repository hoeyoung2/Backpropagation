#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>

#define InputUnitNo     2          // �Է��� ��� �� 
#define HiddenUnitNo    3          // ������ ��� ��
#define OutputUnitNo    1           // ����� ��� ��
#define MaxPatternNo    4          // �н� ���� �ִ��

#define Eta             0.85        // �н���
#define Alpha           0.24         // ���� ����ġ�� ���� ����
#define ErrorFunc       0.03        // �н����� Error
#define Wmin            -0.30       // �ʱ� ����ġ �ּҰ�
#define Wmax            0.30        // �ʱ� ����ġ �ִ밪

/* Sigmoid & Random Numbers */
#define f(x) (1/(1+exp(-(x))))                              // Simoid Function
#define rnd() ((double)rand()/0x7fff * (Wmax-Wmin)+Wmin)   // Random Nubers


/* ��Ʈ��ũ ���� */
double O1[MaxPatternNo][InputUnitNo];                       // �Է��� ��� ��°�
double O2[HiddenUnitNo];                                    // ������ ��� ��°� 
double O3[OutputUnitNo];                                    // ����� ��� ��°�
double t[MaxPatternNo][OutputUnitNo];                       // �����ȣ
double W21[HiddenUnitNo][InputUnitNo];                      // ����ġ (�Է��� -> ������)
double dW21[HiddenUnitNo][InputUnitNo];                     // ����ġ (�Է��� -> ������) Feed Back
double W32[OutputUnitNo][HiddenUnitNo];                     // ����ġ (������ -> �����)
double dW32[OutputUnitNo][HiddenUnitNo];                    // ����ġ (������ -> �����) Feed Back
double bias2[HiddenUnitNo];                                 // ������ ��� Ȱ��ȭ
double dbias2[HiddenUnitNo];                                // ������ ������ ��� Ȱ��ȭ
double bias3[OutputUnitNo];                                 // ����� ��� Ȱ��ȭ 
double dbias3[OutputUnitNo];                                // ����� ������ ��� Ȱ��ȭ
int learning_pattern_no;                                    // �н� ���� ��
int test_pattern_no;                                        // �н� ��, �׽�Ʈ ���� ��

/* ���� ���α׷� */
void main(int argc, char* argv[])
{
    int i, j, k;                                            // ��������(�Է�, ����, ��� �� �ݺ�)
    char filename[30];                                      // ������ �����̸�
    char ss[80];                                            // ���� ó�� ������
    double errorfunc;                                       // �н� ���Ḧ ���� ����

    int num = 0;                                            // Error �ݺ� Ƚ�� 

    /* �� ���α׷��� �ܼ�ȭ �ϱ� ���� �Լ��� �и� */
    void propagation();                                     // ������ ��Ʈ��ũ �Լ�
    void back_propagation();                                // ������ ��Ʈ��ũ �Լ�
    void state();                                           // ��°� ǥ�� �Լ�
    void read_file();                                       // ���� �б� �Լ�
    void initialize();                                      // ����ġ �ʱ�ȭ �Լ�

    printf("Learning File : ");
    scanf("%s", &filename);                                
    printf("\n # �Է��� ���ϸ�: %s", filename);

    /* ����ġ��ȭ ���� ���� ����*/
    FILE* fp = fopen("ErrorFunc.txt", "w");

    read_file(filename);                         
    initialize();       

    printf("\n\n ****************** �н��ϱ� �� ******************\n");
    printf("\tPattern Output1 Output2 Output3 Output4\n");

    /* �н� ���Ͽ� ���� �н� �� ����� ǥ��*/
    for (errorfunc = 0.0, i = 0; i < learning_pattern_no; i++) {
        state(i);
        for (j = 0; j < OutputUnitNo; j++)
            errorfunc += pow(t[i][j] - O3[j], 2.0);
    }
    errorfunc /= 2;
    printf(" ErrorFunc : %.3f\n", errorfunc);

    /* �н� ���� */
    printf("\n******************* �н� ���� ******************* ");
    printf("\nCount Pattern  Output1  Output2  Output3  Output4\n");
    for (i = 0; errorfunc > ErrorFunc; )                                 // Error �� ���� �н�����
    {
        for (j = 0; j < learning_pattern_no; j++){                      // �н� ������ ��
            propagation(j);                                             // ������ �н�
            back_propagation(j);                                        // ������ �н�
        }
        for (errorfunc = 0.0, j = 0; j < learning_pattern_no; j++){        // �н� ������ ��
            printf("%d", ++i);
            state(j);                                                   // state() �Լ�
            for (k = 0; k < OutputUnitNo; k++)                         // ��� ��� �� 
             errorfunc += pow(t[j][k] - O3[k], 2.0);                 // ��ǥ�� - ��°��� ���� Error �� ����
        }
        ++num;
        errorfunc /= 2;
        printf("ErrorFunc : %.3f\n", errorfunc);                        // Error �� ����Ϳ� ���
        fprintf(fp, "%d : %lf \n", num, errorfunc);                      // Error�� ���� �ݺ� �� �� ����
    }
    fclose(fp);                                                         // ���� ���� �ݱ�

    /* �н� ���� */
    printf("\n *************** �ű� ������ ��� ***************\n");
    printf("\tPattern Output1 Output2 Output3 Output4\n");

    /* �ű� ������ ���� ��� */
    for (i = learning_pattern_no; i < learning_pattern_no + test_pattern_no; i++)        // �ű� ������ �� ��ŭ ����
        state(i);
}

/* ������ �Է������� ��������� */
void propagation(p)
int p;
{
    int i, j;
    double net;

    /* ������ ��忡�� ��� ��� */
    for (i = 0; i < HiddenUnitNo; i++){
        for (net = 0.0, j = 0; j < InputUnitNo; j++) {
            net += W21[i][j] * O1[p][j];            // W21 ����ġ�� O1�� ��
        }         
         O2[i] = f(net + bias2[i]);                                      // O2 ��°� = Ȱ��ȭ�Լ� f()
    }

    /* ��� ��忡�� ��� ��� */
    for (size_t i = 0; i < OutputUnitNo; i++)
    {
        for (net = 0.0, j = 0; j < HiddenUnitNo; j++) {
            net += W32[i][j] * O2[j];                               // W32 ����ġ�� O2�� ��
        }
        O3[i] = f(net + bias3[i]);                                  // O3 ��°� = Ȱ��ȭ �Լ� f()
    }
}

/* ������ ����ġ ������ ���� ��������� �Է������� */
int p;
void back_propagation(p){
    int i, j;
    double d2[HiddenUnitNo];                                            // ����ġ ���� �Է� -> ����
    double d3[OutputUnitNo];                                            // ����ġ ���� ���� -> ���
    double sum;                                                         // �ո� ���ϱ� ���� ����

/* �����������: dj=(oj-yj) f'(i,j) */
    for (i = 0; i < OutputUnitNo; i++) {
        d3[i] = (t[p][i] - O3[i]) * O3[i] * (1 - O3[i]);     // ��� ���� ��
    }                   

    /* ����ġ ����  w_(i,j) =  ��d_(j) O_(i)*/
    for (i = 0; i < HiddenUnitNo; i++)
    {
        for (sum = 0.0, j = 0; j < OutputUnitNo; j++)
        {
            dW32[j][i] = Eta * d3[j] * O2[i] + Alpha * dW32[j][i];      // Eta �н����� ������ ����ġ�� Alpha �н��� �߰�
            W32[j][i] += dW32[j][i];                                    // ������ �н� ����ġ�� ������ ����ġ�� ��ȯ 
            sum += d3[j] * W32[j][i];                                   // �� = ����� ���̾�� ����ġ(�����->������)
        }
        /* d_(j) = (�� w_(j,i) d_(i)) f'(i,j) */
        d2[i] = O2[i] * (1 - O2[i]) * sum;
    }
    for (i = 0; i < OutputUnitNo; i++)
    {
        dbias3[i] = Eta * d3[i] + Alpha * dbias3[i];                    // Eta �н����� ������ ����ġ�� Alpha �н��� �߰�
        bias3[i] += dbias3[i];                                          // ��� ��� Ȱ��ȭ
    }

    /* ?w_(i,j) =  ��d_(j) O_(i) */
    for (i = 0; i < InputUnitNo; i++)
    {
        for (j = 0; j < HiddenUnitNo; j++)
        {
            dW21[j][i] = Eta * d2[j] * O1[p][i] + Alpha * dW21[j][i];   // ������ ������ ����ġ
            W21[j][i] += dW21[j][i];                                    // W21 ����ġ
        }
    }
    for (i = 0; i < HiddenUnitNo; i++)
    {
        dbias2[i] = Eta * d2[i] + Alpha * dbias2[i];                    // �� ���� ������ ��� Ȱ��ȭ
        bias2[i] += dbias2[i];                                          // ������ ��� Ȱ��ȭ
    }
}

/* ��� ���� ǥ�� */
void state(int p){
    int i;
    printf("\t%d -> ", p + 1);                                            // state()�Լ��� ���ԵǴ� p�� + 1 �� 
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

    /* �׽�Ʈ ������ �о� ���� */
    fscanf(fp, "%d", &test_pattern_no);
    for (i = learning_pattern_no; i < learning_pattern_no + test_pattern_no; i++) {
        for (j = 0; j < InputUnitNo; j++) {
            fscanf(fp, "%lf", &O1[i][j]); }
    }
    fclose(fp);
}

/* ����ġ �ʱ�ȭ */
void initialize(){
    int i, j;

    /* ����ġ �ʱ�ȭ �Է��� -> ������ */
    for (i = 0; i < HiddenUnitNo; i++) {
        for (j = 0; j < InputUnitNo; j++) {
            W21[i][j] = ((double)rand() / 0x7fff * (Wmax - Wmin) + Wmin); }
    }

    /* ����ġ �ʱ�ȭ ������ -> ����� */
    for (i = 0; i < OutputUnitNo; i++)  {
        for (j = 0; j < HiddenUnitNo; j++) {
            W32[i][j] = ((double)rand() / 0x7fff * (Wmax - Wmin) + Wmin); }
    }
}