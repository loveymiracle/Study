// Day 22 dp, 수학, 조건문, 배열 > 유한소수 판별하기
package Day22;

import java.util.*;

// 문제 설명
// 소수점 아래 숫자가 계속되지 않고 유한개인 소수를 유한소수라고 합니다.
// 분수를 소수로 고칠 때 유한소수로 나타낼 수 있는 분수인지 판별하려고 합니다. 
// 유한소수가 되기 위한 분수의 조건은 다음과 같습니다.

// 기약분수로 나타내었을 때, 분모의 소인수가 2와 5만 존재해야 합니다.
// 두 정수 a와 b가 매개변수로 주어질 때, a/b가 유한소수이면 1을, 무한소수라면 2를
// return하도록 solution 함수를 완성해주세요.

// 제한사항
// a, b는 정수
// 0 < a ≤ 1,000
// 0 < b ≤ 1,000

// 입출력 예
// a	b	result
// 7	20	1
// 11	22	1
// 12	21	2

public class Solution4 {
	
	public int solution(int a, int b) {
        int answer = 2;
        int gcd = gcd(a, b);
        b /= gcd;
        System.out.println("b = " + b);
        
        while(b != 1) {
        	if(b % 2 == 0) {
        		b /= 2;
        	} else if(b % 5 == 0) {
        		b /= 5;
        	} else {
        		return answer;
        	}
        }
        return answer = 1;
    }
	
	private int gcd(int a, int b) {
		if(b == 0) {
			return a;
		}
		return gcd(b, a % b);
	}
	
	public static void main(String[] args) {
		int a = 11;
		int b = 22;
		int result = new Solution4().solution(a, b);
		System.out.println(result);
	}

}
