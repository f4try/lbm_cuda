#include <iostream>
#include <fstream>
using namespace std;
int main() {
	int img[256][256];
	ifstream imgfile("ketton.txt");
	if (!imgfile.is_open())
	{
		cout << "can not open this file" << endl;
		return 0;
	}
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{	
			float buf;
			imgfile >> buf;
			img[i][j] = (int)buf;
			if (j % 20 == 0) {
				cout << img[i][j] << ",";
			}
			
		}
		cout << endl;
	}
	
}