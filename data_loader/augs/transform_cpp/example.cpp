#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <cstdlib>

#include "ndarray_converter.h"

namespace py = pybind11;


cv::Mat ShiftTransform(cv::Mat cv_im, int d, int di)
{
    cv_im.convertTo(cv_im, CV_32FC1);
    if ((d <= 0) || (di == 0)) { return cv_im; }

    float h = cv_im.rows, w = cv_im.cols;
    float r = ((w/2)*(w/2) + d*d)/(2*d);

    auto s = [di, r, d, w] (int x) -> int {
        if (di == -1) {
            return (int)(sqrt(r*r - pow(x-w/2, 2)) + d - r);
        }
        else if (di == 1) {
            return (int)(d - (sqrt(r*r - pow(x - w/2, 2)) + d - r));
        }
        else if (di == -2) {
            return (int)(d - d/w*x);
        }
        else {
            return (int)(d/w*x);
        }
    };

    cv::Mat shift_im((int)h + d, (int)w, CV_32FC1, cv::Scalar::all(0.f));

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            //std::cout << s(i) << std::endl;
            *(shift_im.ptr<float>(j+s(i), i)) = *(cv_im.ptr<float>(j, i));
        }
    }
    cv::resize(shift_im, cv_im, cv::Size(int(w), int(h)));

    return cv_im;
}


cv::Mat SPNoise(cv::Mat cv_im, double prob)
{
    cv_im.convertTo(cv_im, CV_32FC1);
    int h = cv_im.rows, w = cv_im.cols;
    double thres = 1. - prob;
    srand((unsigned)time(NULL));
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            double rnd = rand() / (double)RAND_MAX;
            if (rnd < prob) {
                *(cv_im.ptr<float>(i, j)) = 0.f;
            }
            else if (rnd > thres) {
                *(cv_im.ptr<float>(i, j)) = 255.f;
            }
        }
    }

    return cv_im;
}


cv::Mat GaussianNoise(cv::Mat cv_im, float u, float v) 
{
    cv_im.convertTo(cv_im, CV_32FC1);
	if (cv_im.empty() || cv_im.channels() != 1) {
		std::cout << "Image is empty or not 3 channels" << std::endl;
	}

	cv::Mat noiseImg(cv_im.size(), CV_32FC1);        //存放噪声图像
	cv::RNG rng((unsigned)time(NULL));               //生成随机数 （均值，高斯）
	rng.fill(noiseImg, cv::RNG::NORMAL, u, v);       //随机高斯填充矩阵

	cv_im += noiseImg;                               //添加噪声
	cv_im.convertTo(cv_im, CV_8UC1);            //将double转为uchar 自动截断小于零和大于255的值
    cv_im.convertTo(cv_im, CV_32FC1);

    return cv_im;
}


cv::Mat ElasticDeformations(cv::Mat src,
                            float sigma = 4.f,
                            float alpha = 34.f,
                            bool bNorm = false)
{
    int h = src.rows, w = src.cols;
    src.convertTo(src, CV_32FC1);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat dx(src.size(), CV_64FC1);
    cv::Mat dy(src.size(), CV_64FC1);

    double low = -1.0;
    double high = 1.0;

    //The image deformations were created by first generating
    //random displacement fields, that's dx(x,y) = rand(-1, +1) and dy(x,y) = rand(-1, +1)
    cv::randu(dx, cv::Scalar(low), cv::Scalar(high));
    cv::randu(dy, cv::Scalar(low), cv::Scalar(high));

    //The fields dx and dy are then convolved with a Gaussian of standard deviation sigma(in pixels)
    cv::Size kernel_size(sigma*6 + 1, sigma*6 + 1);
    cv::GaussianBlur(dx, dx, kernel_size, sigma);
    cv::GaussianBlur(dy, dy, kernel_size, sigma);

    //If we normalize the displacement field (to a norm of 1,
    //the field is then close to constant, with a random direction
    if (bNorm) {
        dx /= cv::norm(dx, cv::NORM_L1);
        dy /= cv::norm(dy, cv::NORM_L1);
    }

    //The displacement fields are then multiplied by a scaling factor alpha
    //that controls the intensity of the deformation.
    dx *= alpha;
    dy *= alpha;

    //Inverse(or Backward) Mapping to avoid gaps and overlaps.
    cv::Rect checkError(0, 0, w, h);

    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x) {
            int org_x = x - *(dx.ptr<double>(y, x));
            int org_y = y - *(dy.ptr<double>(y, x));
            if(checkError.contains(cv::Point(org_x, org_y))) {
                *(dst.ptr<float>(y, x)) = *(src.ptr<float>(org_y, org_x));
            }
        }
    
    return dst;
}


PYBIND11_MODULE(example, m) {

    NDArrayConverter::init_numpy();

    m.def("shift_cpp", &ShiftTransform, "Do shift transform",
            py::arg("cv_im"), py::arg("d"), py::arg("di"));
    m.def("sp_noise_cpp", &SPNoise, "Add sp noise to image",
          py::arg("cv_im"), py::arg("prob"));
    m.def("gaussian_noise_cpp", &GaussianNoise, "Add gaussian noise to image",
          py::arg("cv_im"), py::arg("u"), py::arg("v"));
    m.def("elastic_cpp", &ElasticDeformations, "Do elastic transform",
          py::arg("src"), py::arg("sigma"), py::arg("alpha"), py::arg("bNorm"));
}
