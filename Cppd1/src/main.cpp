#include <iostream>
#include "../include/addnum.h"

using namespace std;

int main() {
	int a;
	int b;

	int sum;
        cout << "kaunse number jodu?:\n";
	cin >> a;
	cout << "aur dusra number?\n";
	cin >> b;
	sum = add_num(a, b);
	
	cout << "The sum is :"<< sum;
	return 0;
	

}


	

