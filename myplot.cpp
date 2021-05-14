#include<Python.h>
#include<string>
using namespace std;
#include <iostream>

template<class T>
string arr_to_string_list(T* arr, int N) {
	string s = "[";
	for (int i = 0; i < N; ++i) {
		s += to_string(arr[i]);
		if (i != N - 1) s += ",";
	}
	s += "]";
	return s;
}

template<class T, class V = int>
void plot(T* x, int N1, V* y = NULL, bool equal = false) {
	PyRun_SimpleString("import matplotlib.pyplot as plt");
	if (equal) {
		PyRun_SimpleString("plt.axis(\"equal\")");
	}

	string cmd = "plt.plot(";
	string s1 = arr_to_string_list(x, N1);
	if (y != NULL) {
		string s2 = arr_to_string_list(y, N1);
		cmd += (s1 + "," + s2 + ")");
		PyRun_SimpleString(cmd.c_str());
	}
	else {
		cmd += (s1 + ")");
		PyRun_SimpleString(cmd.c_str());
	}
	PyRun_SimpleString("plt.show()");
}

void pythonInitial() {
	Py_Initialize(); /*初始化python解释器,告诉编译器要用的python编译器*/
	string path = ".";
	string chdir_cmd = string("sys.path.append(\"") + path + "\")";
	const char* cstr_cmd = chdir_cmd.c_str();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(cstr_cmd);
}

void main() {
	double a[] = { 1.2, 2.0, 3, 4, 5, 6, 3, 2, 3, 2, 31, 2, 1, 4 };
	pythonInitial();
	plot(a, 14); // 14是数组长度
	Py_Finalize(); /*结束python解释器，释放资源*/
	system("pause");
}