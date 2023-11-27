package hw.poly.view;

import java.util.Scanner;

import hw.poly.controller.Controller;
import hw.poly.model.vo.Book;
import hw.poly.model.vo.Member;

public class LibraryMenu {
	private Controller lc = new Controller();
	private Scanner sc = new Scanner(System.in);
	
	public void mainMenu() {
		System.out.println("이름: ");
		String name = sc.nextLine();
		System.out.println("나이: ");
		int age = Integer.parseInt(sc.nextLine());
		System.out.println("성별 (M/F): ");
		char gender = sc.nextLine().charAt(0);
		
		Member mem = new Member(name, age, gender);
		lc.insertMember(mem);
		
		while(true) {
			System.out.println("===메뉴===\n1. 마이페이지\n2. 도서 전체 조회\n3. 도서 검색\n4. 도서 대여하기\n9. 프로그램 종료하기\n메뉴 번호 :");
			int menuNum = Integer.parseInt(sc.nextLine());
			switch (menuNum) {
				case 1: lc.myInfo(); System.out.println(mem); break;
				case 2: selectAll(); break;
				case 3: searchBook(); break;
				case 4: rentBook(); break;
				case 9: return;
			}
		}
	}
	
	public void selectAll() {
		Book bList[] = lc.selectAll();
		
		for(int i = 0; i < bList.length; i++) {
			System.out.println(i + "번 도서 : " + bList[i]);
		}
	}
	
	public void searchBook() {
		System.out.println("검색할 제목 키워드 : ");
		String keyword = sc.nextLine();
		
		Book searchList[] = lc.searchBook(keyword);
		
		for(int i = 0; i < searchList.length; i++) {
			if(searchList[i] != null) {
				System.out.println(searchList[i]);
			}
		}
	}
	
	public void rentBook() {
		selectAll();
		System.out.println("대여할 도서 번호 선택 : ");
		
		int index = Integer.parseInt(sc.nextLine());
		int result = lc.rentBook(index);
		
		switch (result) {
		case 0: System.out.println("성공적으로 대여되었습니다."); break;
		case 1: System.out.println("나이 제한으로 대여 불가능입니다.");  break;
		case 2: System.out.println("성공적으로 대여되었습니다. 요리학원 쿠폰이 발급되었으니 마이페이지에서 확인하세요."); break;
		}
	}
}