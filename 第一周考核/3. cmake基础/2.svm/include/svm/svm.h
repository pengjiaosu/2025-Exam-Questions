#ifndef SVM_SVM_H_
#define SVM_SVM_H_

#include <vector>
#include <cmath>

#include <Eigen/Core>

namespace svm
{
    using std::vector;
    using Eigen::VectorXf;

    class SVMKernel {
        public:
        SVMKernel() {}
        ~SVMKernel() = default;

        virtual void
        setParameter(float p) {};

        virtual float 
        K(VectorXf x1, VectorXf x2) = 0;
    };


    class SVM {
        public:
        SVM();
        ~SVM() = default;

        /**
         * @brief Set the Dataset object
         * 
         * @param dataset 
         * @param label 
         */
        inline void
        setDataset(const vector<VectorXf> &dataset, const vector<int> &label) { x_ = dataset; y_ = label; }

        /**
         * @brief Set the Kernel Func object
         * 
         * @param kernel 
         */
        inline void
        setKernelFunc(SVMKernel *kernel) { K_ = kernel; }        

        /**
         * @brief Set the Iter Time object
         * 
         * @param t 
         */
        inline void
        setIterTime(int t) { iters_ = t; }

        /**
         * @brief Set the Soft Coefficients object
         * 
         * @param c punish coefficient
         * @param e lossen coefficient
         */
        inline void
        setSoftCoefficients(float c) { C_ = c; }

        vector<float>
        getA() const;

        vector<int>
        getY() const;

        vector<VectorXf>
        getX() const;

        inline float
        getb() const { return b_; }

        bool
        train(bool seed = true);

        bool
        predict(const VectorXf &data, float &label);

        private:
        // use rand seed
        bool withSeed_{true};
        // X
        vector<VectorXf> x_;
        // labels
        vector<int> y_;
        // error cache
        vector<float> eCache_;
        // bias
        float b_{0.f};
        // lagrange function coefficients: alpha
        vector<float> a_;
        // a1_new - a1_old
        float deltaA1_;
        // a2_new - a2_old
        float deltaA2_;
        // K(x1, x2)
        SVMKernel *K_;
        // eta = K_1,1 + K_2,2 - 2 * K_1,2
        float n_{0.f};
        // soft gap: 1/2 * ||w||^2 + C * epsilon
        // pubishment coeff
        float C_{10000};
        // lossen param
        float e_{0.01};
        // iteration times
        int iters_{50};
        // parameters changed
        bool isChanged{true};
        // first time flag
        bool isFirst{true};

        // f(x) = sigma(a_i * y_i * x_i * x) + b
        inline float 
        f_(const VectorXf &x) {
            float wSum = 0.f;
            for (int i = 0; i < y_.size(); i++) {
                wSum += a_[i] * (float)y_[i] * K_->K(x_[i], x);
            }
            return (wSum + b_);
        }

        // SVM init
        bool
        init();

        // select j
        int
        select(int i);

        // SMO algorithm
        void 
        SMO(int &count);

        // KKT
        bool 
        isKKT(int i);

        // clip alpha
        void
        clip(int i, float L, float H);

        // update eCache
        void
        updateErrCache();

        // update bias
        void 
        updateBias(int i, int j);
    };

    class LinearKernel: public SVMKernel {
        public:
        inline float
        K(VectorXf x1, VectorXf x2) override {return x1.transpose() * x2; }
    };

    class PolynomialKernel: public SVMKernel {
        public:
        inline void
        setParameter(float p) { p_ = p; }

        inline float 
        K(VectorXf x1, VectorXf x2) override { return pow((x1.transpose() * x2 + 1), p_); }

        private:

        int p_{2};
    };

    class GaussianKernel: public SVMKernel {
        public:

        inline void
        setParameter(float p) { sigma_ = p; }

        inline float 
        K(VectorXf x1, VectorXf x2) override { return exp(-pow((x1 - x2).norm(), 2) / (2 * sigma_ * sigma_)); }

        private:
        // G(x, z) = exp(-(x - z) ^ 2 / (2 * sigma ^ 2))
        float sigma_{0.1};
    };
} // namespace svm

#endif