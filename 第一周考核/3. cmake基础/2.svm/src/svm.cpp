#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "svm/svm.h"

#include <Eigen/Core>

namespace svm
{
    SVM::SVM() {
    }

    bool
    SVM::isKKT(int i) {
        // a = 0 && y * u > 1
        // a = C && y * u < 1
        // 0 < a < C && y * u = 1
        // e_ = std::max(0.f, 1 - y_[i] * f_(x_[i]));
        float Ei = eCache_[i];
        if ((y_[i] * Ei <= -1e-5  and a_[i] < C_) or (y_[i] * Ei >= 1e-5 and a_[i] > 0)
            or (fabs(y_[i] * Ei) < 1e-5 and (fabs(a_[i] < 1e-5) or fabs(a_[i] - C_) < 1e-5)))  {
            return false;
        }
        return true;
    }

    void
    SVM::updateBias(int i, int j) {
        Eigen::Matrix2f kMat;
        Eigen::Vector2f deltaA(deltaA1_, deltaA2_), coeff(0, 0);
        kMat(0, 0) = K_->K(x_[i], x_[i]) * y_[i];
        kMat(0, 1) = K_->K(x_[i], x_[j]) * y_[j];
        kMat(1, 0) = K_->K(x_[i], x_[j]) * y_[i];
        kMat(1, 1) = K_->K(x_[j], x_[j]) * y_[j];
        coeff = kMat * deltaA;
        float E1, E2;
        E1 = eCache_[i], E2 = eCache_[j];
        float b1New = -E1 - coeff(0) + b_, b2New = -E2 - coeff(1) + b_;
        if (a_[i] > 1e-5 and a_[i] < C_) {
            b_ = b1New;
        }
        else if (a_[j] > 1e-5 and a_[j] < C_) {
            b_ = b2New;
        }
        else {
            b_ = (b1New + b2New) / 2.f;
        }
    }

    void
    SVM::updateErrCache() {
        for (int i = 0; i < eCache_.size(); i++) {
            eCache_[i] = f_(x_[i]) - y_[i];
            // std::cout << "err: " << eCache_[i] << std::endl;
        }
    }

    int 
    SVM::select(int i) {
        int j = 0;
        if (isFirst) {
            j = i;
            while (j == i) {
                j = rand() % y_.size();
            }
        }
        else {
            float maxErr = 0.f;
            int count = 0;
            for (auto && err : eCache_) {
                if (a_[count] == 0 or count == i) {
                    count++;
                    continue;
                }
                if (fabs(err - eCache_[i]) > maxErr) {
                    maxErr = fabs(err - eCache_[i]);
                    j = count;
                }
                count++;
            }
        }
        return j;
    }

    bool
    SVM::init() {
        if (x_.empty() or y_.empty()) {
            std::cout << "No data input!\n";
            return false;
        }

        // init eCache
        eCache_.clear();
        eCache_.resize(y_.size());

        // init a and w
        a_.clear();
        a_.reserve(y_.size());
        for (int i = 0; i < y_.size(); i++) {
            a_.push_back(0);
        }
        deltaA1_ = 0;
        deltaA2_ = 0;
        b_ = 0;
        isChanged = true;
        isFirst = true;
        return true;
    }

    void
    SVM::clip(int i, float L, float H) {
        if (a_[i] < L) {
            a_[i] = L;
        }
        else if (a_[i] > H) {
            a_[i] = H;
        }
    }
    
    void
    SVM::SMO(int &count) {
        int num = y_.size();
        int i = 0;
        int unchangeTimes = 0;
        while (i < num) {
            if ((isFirst and not isKKT(i)) or (not isFirst and a_[i] < C_ and a_[i] > 0)) {
                int j = select(i);
                float old_a1 = a_[i];
                float old_a2 = a_[j];
                
                n_ = K_->K(x_[i], x_[i]) + K_->K(x_[j], x_[j]) - 2 * K_->K(x_[i], x_[j]);
                if (n_ <= 0) {
                    std::cout << "eta <= 0" << '\n';
                    i++;
                    unchangeTimes++;
                    continue;
                }
                float a1Err = 0.f, a2Err = 0.f;
                a1Err = eCache_[i];
                a2Err = eCache_[j];
                a_[j] += (float)y_[j] * (a1Err - a2Err) / n_;
                float L = 0, H = 0;
                if (y_[i] != y_[j]) {
                    L = std::max(0.f, old_a2 - old_a1);
                    H = std::min(C_, C_ + old_a2 - old_a1);
                }
                else {
                    L = std::max(0.f, old_a1 + old_a2 - C_);
                    H = std::min(C_, old_a1 + old_a2);
                }
                clip(j, L, H);
                if (fabs(a_[j] - old_a2) < 1e-5) {
                    i++;
                    unchangeTimes++;
                    continue;
                }
                a_[i] -= (float)y_[i] * (float)y_[j] * (a_[j] - old_a2);
                deltaA1_ = a_[i] - old_a1;
                deltaA2_ = a_[j] - old_a2;
                updateBias(i, j);
                updateErrCache();
                i++;
            }
            else {
                i++;
                unchangeTimes++;
                continue;
            }
        }
        if (unchangeTimes == y_.size()) {
            isChanged = false;
        }
        count++;
    }

    bool
    SVM::train(bool seed) {
        int epochs = 0;
        bool notRight = true;
        withSeed_ = seed;
        if (withSeed_) srand(time(NULL));
        else srand(100);
        while (epochs < 1000) {
            if (!init()) {
                return false;
            }
            updateErrCache();
            int count = 0;
            while (count < iters_) {
                SMO(count);
                // if (count % 1 == 0) {
                //     std::cout << "iterations: " << count << std::endl;
                // }
                if (!isChanged) {
                    if (isFirst) {
                        break;
                    }
                    isFirst = true;
                    continue;
                }
                isFirst = false;
            }
            int accuracteNum = 0;
            for (int i = 0; i < y_.size(); i++) {
                float label = 0.f;
                predict(x_[i], label);
                if (label * y_[i] > 0) {
                    accuracteNum++;
                }
            }
            std::cout << "epochs: " << epochs << '\n';

            // project (0.5, 1) to (0, 1)
            float accuracy = log2((float)accuracteNum / (float)y_.size() * 2.f);
            if (accuracy >= 0.8) {
                notRight = false;
                std::cout << "Train dataset accuracy: " << accuracy << std::endl;
                break;
            }
            // else {
            //     std::cout << "Train dataset accuracy: " << accuracy << std::endl;
            // }
            epochs++;
        }   
        if (not notRight) {
            return true;
        }
        else {
            std::cout << "Err: Fail to find proper model!" << std::endl;
            return false;
        }
    }

    bool
    SVM::predict(const VectorXf &data, float &label) {
        if (data.size() != x_[0].size()) {
            std::cout << "data size wrong!\n";
            return false;
        }
        for (int i = 0; i < y_.size(); i++) {
            if (a_[i] > 0 and a_[i] < C_)
                label += a_[i] * y_[i] * K_->K(x_[i], data);
        }
        label += b_;
        return true;
    }

    vector<float>
    SVM::getA() const {
        vector<float> a;
        a.clear();
        for (int i = 0; i < a_.size(); i++) {
            if (a_[i] > 0 and a_[i] < C_) {
                a.push_back(a_[i]);
            }
        }
        return a;
    }

    vector<int>
    SVM::getY() const {
        vector<int> y;
        y.clear();
        for (int i = 0; i < a_.size(); i++) {
            if (a_[i] > 0 and a_[i] < C_) {
                y.push_back(y_[i]);
            }
        }
        return y;
    }

    vector<VectorXf>
    SVM::getX() const {
        vector<VectorXf> x;
        x.clear();
        for (int i = 0; i < a_.size(); i++) {
            if (a_[i] > 0 and a_[i] < C_) {
                x.push_back(x_[i]);
            }
        }
        return x;
    }

} // namespace svm