#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <math.h>
#define PI 3.1415927
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}
void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}
const char* vertexShaderSource = "#version 330 core\n"
"layout (location =0) in vec3 aPos;\n"
"layout (location =1) in vec3 aColor;\n"
"out vec3 ourColor;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(aPos,1.0);\n"
//"	vertexColor = vec4(0.5,0.0,0.0,1.0);\n"
	"ourColor = aColor;\n"
"}\0";
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
//"uniform vec4 ourColor;\n"
"void main(){\n"
"	FragColor=vec4(ourColor,1.0f);\n"
"}\0";
int main() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(800, 600, "shader", NULL, NULL); 
	glfwMakeContextCurrent(window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glViewport(0, 0, 800, 600);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	//int nrAttributes;
	//glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &nrAttributes);
	//std::cout << "Maximum vertex attributes:" << nrAttributes << std::endl;
	float vertices[] = { 0.5,-0.5,0,
						1.0f,0.0f,0.0f,
						0.5,0.5,0,
					    0.0f,1.0f,0.0f,
						-0.5,0.5,0,
						0.0f,0.0f,1.0f, };
	unsigned int VBO;
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);
	
	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	unsigned int shaderProgram;
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	/*float timeValue = glfwGetTime();
	float greenValue = (sin(timeValue / 2.0f)) + 0.5f;
	int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
	glUseProgram(shaderProgram);
	glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);*/
	glUseProgram(shaderProgram);

	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	//glEnableVertexAttribArray(0);
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		float timeValue = glfwGetTime();
		float redValue = (sin(timeValue / 5.0f)) + 0.5f;
		float greenValue = (sin(timeValue / 3.0f)) + 0.5f;
		float blueValue = (cos(timeValue / 2.0f)) + 0.5f;
		//int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
		glUseProgram(shaderProgram);
		//glUniform4f(vertexColorLocation, redValue, greenValue, blueValue, 1.0f);
		glBindVertexArray(VAO);
		for (int i = 0; i < 3; i++) {
			vertices[i * 6] = 0.5 * cos(timeValue + i * 2 * PI / 3);
			vertices[i * 6 + 1] = 0.5 * sin(timeValue + i * 2 * PI / 3);
			switch(i){
			case 0: {
				vertices[i * 6 + 3] = (sin(timeValue / 1.0f)) + 0.5f;;
				vertices[i * 6 + 4] = (sin(timeValue / 2.0f)) + 0.5f;;
				vertices[i * 6 + 5] = (cos(timeValue / 3.0f)) + 0.5f;;
				break;
			}
			case 1: {
				vertices[i * 6 + 3] = (cos(timeValue / 2.0f)) + 0.5f;;
				vertices[i * 6 + 4] = (cos(timeValue / 1.0f)) + 0.5f;;
				vertices[i * 6 + 5] = (sin(timeValue / 3.0f)) + 0.5f;;
				break;
			}
			case 2: {
				vertices[i * 6 + 3] = (cos(timeValue / 3.0f)) + 0.5f;;
				vertices[i * 6 + 4] = (sin(timeValue / 2.0f)) + 0.5f;;
				vertices[i * 6 + 5] = (sin(timeValue / 1.0f)) + 0.5f;;
				break;
			}
			}
		}
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	glfwTerminate();
	return 0;
}