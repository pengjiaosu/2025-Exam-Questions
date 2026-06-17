#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

#include <Eigen/Core>

#include <svm/svm.h>

using std::string;
using std::ios;
using std::cout;
using std::vector;
using std::pair;

void 
spilt(const string &line, const string &c, vector<string> &v)
{
	int pos1 = 0;
	int pos2 = line.find(c);

	while(pos2 < line.size() and pos2 >= 0)
	{
		v.push_back(line.substr(pos1, pos2 - pos1));
		pos1 = pos2 + c.size();
		pos2 = line.find(c, pos1);
	}
    v.push_back(line.substr(pos1));
    // substr有2种用法： 假设：string s = "0123456789"; string sub1 = s.substr(5); 
    // 只有一个数字5表示从下标为5开始一直到结尾：sub1 = "56789" string sub2 = s.substr(5, 3); 
    // 从下标为5开始截取长度为3位：sub2 = "567" 
}

template<typename T> T 
StringToNum(const string &s)
{
	std::istringstream iss(s);
	T num;
	iss>>num;
	return num;
}

vector<pair<Eigen::VectorXf, string>>
readCSV(std::ifstream &fin, const string &filePath, const string &split, bool head = true) {
    fin.open(filePath, ios::in);
    vector<pair<Eigen::VectorXf, string>> result;
    result.clear();
    if (fin.fail()) {
        cout << "No such a file name " << filePath << '\n';
        return result;
    }
    string line;

    vector<vector<string>> dataset;
    vector<string> labels;
    labels.reserve(dataset.size());
    int num = head ? -1 : 0;
    while (getline(fin, line) and fin.good()) {
        if (num < 0) {
            num++;
            continue;
        }
        dataset.push_back(vector<string>());
        spilt(line, split, dataset[num]);
        labels.push_back(*(dataset[num].end() - 1));
        dataset[num].pop_back();
        dataset[num].pop_back();    // kick off the second last num
        num++;
    }
    result.reserve(dataset.size());
    for (int i = 0; i < dataset.size(); i++) {
        Eigen::VectorXf data;
        string label = labels[i];
        data = Eigen::Matrix<float, Eigen::Dynamic, 1>();
        data.resize(dataset[i].size(), 1);
        for (int j = 0; j < dataset[i].size(); j++) {
            data(j) = StringToNum<float>(dataset[i][j]);
        }
        result.push_back(pair<Eigen::VectorXf, string>(data, label));
    }
    return result;
}

void shuffle(vector<pair<Eigen::VectorXf, string>> &dataset) {
    srand(time(NULL));
    std::vector<int> checkArray;
    checkArray.resize(100);
    for (auto && elem : checkArray) {
        elem = 0;
    }
    vector<pair<Eigen::VectorXf, string>> temp;
    temp.reserve(100);
    for (int k = 0; k < 100; k++) {
        int index = rand() % 100;
        while (checkArray[index] == 1)
        {
            index = rand() % 100;
        }
        checkArray[index] = 1;
        temp.push_back(dataset[index]);
    }
    dataset = temp;
}

int main() {
    std::ofstream fout;
    string filePath("./data/iris.csv");
    std::ifstream fin;
    vector<pair<Eigen::VectorXf, string>> dataset = readCSV(fin, filePath, ",", false);
    shuffle(dataset);
    vector<Eigen::VectorXf> trainset;
    vector<int> trainlabels;
    vector<Eigen::VectorXf> testset;
    vector<int> testlabels;
    trainset.reserve(80);
    trainlabels.reserve(80);
    testset.reserve(20);
    testlabels.reserve(20);
    for (int i = 0; i < 80; i++) {
        trainset.push_back(dataset[i].first);
        if (dataset[i].second == "setosa")
            trainlabels.push_back(1);
        else if (dataset[i].second == "versicolor")
            trainlabels.push_back(-1);
    }
    for (int j = 0; j < 20; j++) {
        testset.push_back(dataset[j + 80].first);
        if (dataset[j + 80].second == "setosa")
            testlabels.push_back(1);
        else if (dataset[j + 80].second == "versicolor")
            testlabels.push_back(-1);
    }
    svm::GaussianKernel *kernel = new svm::GaussianKernel();
    kernel->setParameter(0.1);
    // svm::LinearKernel *kernel = new svm::LinearKernel();
    svm::SVM mySvm;
    mySvm.setDataset(trainset, trainlabels);
    mySvm.setIterTime(100);
    mySvm.setKernelFunc(kernel);
    mySvm.setSoftCoefficients(1000);
    mySvm.train();
    vector<Eigen::VectorXf> x = mySvm.getX();
    vector<float> a = mySvm.getA();
    vector<int> y = mySvm.getY();
    // fout.open("./block_param/iris.txt", std::ios_base::out);
    // fout << "RBF" << '\n';
    // for (int i = 0; i < a.size(); i++) {
    //     for (int j = 0; j < x[i].size(); j++) {
    //         fout << x[i](j) << '\n';
    //     }
    //     fout <<  a[i] << "\n" << y[i] << '\n';
    // }
    // float b = mySvm.getb();
    // fout << b << '\n';
    // fout.close();
    float accurateNum = 0;
    for (int i = 0; i < testlabels.size(); i++) {
        float label = 0;
        mySvm.predict(testset[i], label);
        std::cout << "predicted label: ";
        if (label < 0) {
            cout << "versicolor; \n";
        }
        else {
            cout << "setosa; \n";
        }
        std::cout << "original label: ";
        if (testlabels[i] < 0) {
            cout << "versicolor; \n";
        }
        else {
            cout << "setosa; \n";
        }
        if (label * testlabels[i] > 0) {
            accurateNum += 1;
        }
    }
    float accuracy = log2((float)accurateNum / (float)testlabels.size() * 2.f);
    cout << "test accuracy: " << accuracy << std::endl;
    delete kernel;
    return 0;
}
